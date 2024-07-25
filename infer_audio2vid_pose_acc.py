import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import av
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic_pose_acc import AudioPose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid, crop_and_pad
import sys
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN
from src.utils.draw_utils import FaceMeshVisualizer
import pickle
from src.utils.motion_utils import motion_sync
from src.utils.mp_utils  import LMKExtractor
from src.utils.img_utils import pil_to_cv2, cv2_to_pil, center_crop_cv2, pils_from_video, save_videos_from_pils, save_video_from_cv2_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/animation_pose_acc.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    parser.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)
    parser.add_argument("--crop_face", type=int, default=1)
    parser.add_argument("--motion_sync", type=int, default=1)
    parser.add_argument("--paste_back", type=int, default=0)

    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args

def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    ############# model_init started #############

    ## vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    ## denoising net init
    if os.path.exists(config.motion_module_path):
        ### stage1 + stage2
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        ### only stage1
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )

    ## face locator init
    face_locator = FaceLocator(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    visualizer = FaceMeshVisualizer(draw_iris=False, draw_mouse=False)

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    ### load face detector params
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

    ############# model_init finished #############

    width, height = args.W, args.H
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = AudioPose2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    )

    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"
    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    for ref_image_path in config["test_cases"].keys():
        for file_path in config["test_cases"][ref_image_path]:
            if ".wav" in file_path:
                audio_path = file_path
            else:
                pose_dir = file_path

        if args.seed is not None and args.seed > -1:
            generator = torch.manual_seed(args.seed)
        else:
            generator = torch.manual_seed(random.randint(100, 1000000))

        ref_name = Path(ref_image_path).stem
        audio_name = Path(audio_path).stem
        final_fps = args.fps

        # ref_image_pil = Image.open(ref_image_path).convert("RGB")
        #### face crop
        ori_img_pil = Image.open(ref_image_path).convert("RGB")
        face_img = cv2.imread(ref_image_path)
        ori_size = (face_img.shape[1], face_img.shape[0])

        if args.crop_face:
            det_bboxes, probs = face_detector.detect(face_img)
            select_bbox = select_face(det_bboxes, probs)
            if select_bbox is not None:
                xyxy = select_bbox[:4]
                xyxy = np.round(xyxy).astype('int')
                rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]

                r_pad_crop = int((re - rb) * args.facecrop_dilation_ratio)
                c_pad_crop = int((ce - cb) * args.facecrop_dilation_ratio)
                crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + c_pad_crop, face_img.shape[0])]
                print(crop_rect)
                face_img, ori_face_rect = crop_and_pad(face_img, crop_rect)
                print(ori_face_rect)
                ori_face_size = (ori_face_rect[2]-ori_face_rect[0], ori_face_rect[3]-ori_face_rect[1])
                face_img = cv2.resize(face_img, (args.W, args.H))
        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])

        if args.motion_sync:
            if os.path.isdir(pose_dir) or pose_dir[-4:]!='.mp4':
                print("motion_sync skipped. Please replace the pose dir with the driven video to enable it.")
            else:
                imsize = (args.W, args.H)
                ref_image_pil.save('tmp_ref_img.png')
                lmk_extractor = LMKExtractor()

                input_frames_cv2 = [cv2.resize(center_crop_cv2(pil_to_cv2(i)), imsize) for i in pils_from_video(pose_dir)]
                ref_frame = face_img
                ref_det = lmk_extractor(ref_frame)

                sequence_driver_det = []
                try: 
                    for frame in input_frames_cv2:
                        result = lmk_extractor(frame)
                        assert result is not None, "{}, bad video, face not detected".format(driver_video)
                        sequence_driver_det.append(result)
                except:
                    print("motion_sync error: face detection failed")
                    exit()
                pose_save_dir = './{}'.format(ref_image_path.split('/')[-1].replace('.png', ''))
                os.makedirs(pose_save_dir, exist_ok=True)
                sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
                for i in range(len(sequence_det_ms)):
                    with open('{}/{}.pkl'.format(pose_save_dir, i), 'wb') as file:
                        pickle.dump(sequence_det_ms[i], file)
                pose_dir = pose_save_dir


        # ==================== face_locator =====================
        pose_list = []
        for index in range(len(os.listdir(pose_dir))):
            tgt_musk_path = os.path.join(pose_dir, f"{index}.pkl")

            with open(tgt_musk_path, "rb") as f:
                tgt_kpts = pickle.load(f)
            tgt_musk = visualizer.draw_landmarks((args.W, args.H), tgt_kpts)
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk).astype(np.uint8)).convert('RGB')
            pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device="cuda").permute(2,0,1) / 255.0)
        face_mask_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        #face_mask_tensor = torch.zeros_like(face_mask_tensor)


        video = pipe(
            ref_image_pil,
            audio_path,
            face_mask_tensor,
            width,
            height,
            args.L,
            args.steps,
            args.cfg,
            generator=generator,
            audio_sample_rate=args.sample_rate,
            context_frames=12,
            fps=final_fps,
            context_overlap=3
        ).videos

        final_length = min(video.shape[2], face_mask_tensor.shape[2], args.L)
        face_tensor = video[:, :, :final_length, :, :].contiguous()
        video = torch.cat([face_tensor, face_mask_tensor[:, :, :final_length, :, :].detach().cpu()], dim=-1)
        save_videos_grid(
            video,
            f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
            n_rows=2,
            fps=final_fps,
        )

        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4")
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4", codec="libx264", audio_codec="aac")
        print(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4")

        if args.paste_back:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # face_tensor = video[:, :, :final_length, :, :]
            b, c, f, h, w = face_tensor.shape
            reshaped_tensor = face_tensor.view(b*f*c, 1, h, w)
            new_w, new_h = ori_face_size
            resized_tensor = F.interpolate(reshaped_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            review_face_tensor = resized_tensor.view(b, c, f, new_h, new_w)

            mask_img_pil = Image.open('./assets/mask_image_512.png').convert("RGB").resize(ori_face_size)
            mask_img_tensor = transform(mask_img_pil)
            mask_tensor_expanded = mask_img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, final_length, 1, 1, 1).permute(0,2,1,3,4).contiguous()
            b, c, f, h, w = mask_tensor_expanded.shape
            reshaped_tensor = mask_tensor_expanded.view(b*f*c, 1, h, w)
            resized_tensor = F.interpolate(reshaped_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            review_mask_tensor = resized_tensor.view(b, c, f, new_h, new_w)

            ori_img_tensor = transform(ori_img_pil)
            # batch_size = 1
            # num_frames = final_length
            # channels = ori_img_tensor.shape[0]
            # height = ori_img_tensor.shape[1]
            # width = ori_img_tensor.shape[2]
            img_tensor_expanded = ori_img_tensor.unsqueeze(0).unsqueeze(0).repeat(1, final_length, 1, 1, 1).permute(0,2,1,3,4)
            final_mask_tensor = torch.ones_like(img_tensor_expanded)
            final_mask_tensor[:, :, :, ori_face_rect[1]:ori_face_rect[3], ori_face_rect[0]:ori_face_rect[2]] = (1.0 - review_mask_tensor)
            final_face_tensor = torch.zeros_like(img_tensor_expanded)
            final_face_tensor[:, :, :, ori_face_rect[1]:ori_face_rect[3], ori_face_rect[0]:ori_face_rect[2]] = review_face_tensor

            psbk_video = final_mask_tensor * img_tensor_expanded + (1.0 - final_mask_tensor) * final_face_tensor
            save_videos_grid(
                psbk_video,
                f"{save_dir}/psbk_{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
                n_rows=1,
                fps=final_fps,
            )



if __name__ == "__main__":
    main()

