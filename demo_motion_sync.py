from src.utils.mp_utils  import LMKExtractor
from src.utils.draw_utils import FaceMeshVisualizer
from src.utils.img_utils import pil_to_cv2, cv2_to_pil, center_crop_cv2, pils_from_video, save_videos_from_pils, save_video_from_cv2_list
from PIL import Image
import cv2
from IPython import embed
import numpy as np
import copy
from src.utils.motion_utils import motion_sync
import pathlib
import torch
import pickle
from glob import glob
import os

vis = FaceMeshVisualizer(draw_iris=False, draw_mouse=True, draw_eye=True, draw_nose=True, draw_eyebrow=True, draw_pupil=True)

imsize = (512, 512)
visualization = True
driver_video = "./assets/driven_videos/a.mp4"
# driver_videos = glob("/nas2/luoque.lym/evaluation/test_datasets/gt_data/OurDataset/*.mp4")
ref_image = './assets/test_imgs/d.png'
# ref_image = 'panda.png'

lmk_extractor = LMKExtractor()

input_frames_cv2 = [cv2.resize(center_crop_cv2(pil_to_cv2(i)), imsize) for i in pils_from_video(driver_video)]
ref_frame =cv2.resize(cv2.imread(ref_image), (512, 512))
ref_det = lmk_extractor(ref_frame)
# print(ref_det)

sequence_driver_det = []
try: 
    for frame in input_frames_cv2:
        result = lmk_extractor(frame)
        assert result is not None, "{}, bad video, face not detected".format(driver_video)
        sequence_driver_det.append(result)
except:
    print("face detection failed")
    exit()
print(len(sequence_driver_det))

if visualization:
    pose_frames_driver = [vis.draw_landmarks((512, 512), i["lmks"], normed=True) for i in sequence_driver_det]
    poses_add_driver = [(i * 0.5 + j * 0.5).clip(0,255).astype(np.uint8)  for i, j in zip(input_frames_cv2, pose_frames_driver)]

save_dir = './{}'.format(ref_image.split('/')[-1].replace('.png', ''))
os.makedirs(save_dir, exist_ok=True)

sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
for i in range(len(sequence_det_ms)):
    with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
        pickle.dump(sequence_det_ms[i], file)
if visualization:
    pose_frames = [vis.draw_landmarks((512, 512), i, normed=False) for i in sequence_det_ms]
    poses_add = [(i * 0.5 + ref_frame * 0.5).clip(0,255).astype(np.uint8) for i in pose_frames]

# sequence_det_ms = motion_sync(sequence_driver_det, ref_det, per_landmark_align=False)
# for i in range(len(sequence_det_ms)):
#     tmp = {}
#     tmp["lmks"] = sequence_det_ms[i]
#     with open('{}_v2/{}.pkl'.format(save_dir, i), 'wb') as file:
#         pickle.dump(tmp, file)
# pose_frames_wo_lmkalign = [vis.draw_landmarks((512, 512), i, normed=False) for i in sequence_det_ms]
# poses_add_wo_lmkalign = [(i * 0.5 + ref_frame * 0.5).clip(0,255).astype(np.uint8) for i in pose_frames_wo_lmkalign]

poses_cat = [np.concatenate([i, j], axis=1) for i, j in zip(poses_add_driver, poses_add)]

save_video_from_cv2_list(poses_cat, "./vis_example.mp4", fps=24.0)


# for ref_image in ref_images[:1]:
# # for driver_video in driver_videos:
#     # ref_image = "./samples/007.png"
#     # save_dir = '/nas2/jiajiong.caojiajio/data/test_pose/OurDataset/{}'.format(driver_video.split('/')[-1].replace('.mp4', ''))
#     save_dir = './{}'.format(ref_image.split('/')[-1].replace('.png', ''))
#     os.makedirs(save_dir+'_v1', exist_ok=True)
#     os.makedirs(save_dir+'_v2', exist_ok=True)
#     #"./samples/hedra_003.png"

#     #"./samples/video_temp_fix.mov"
#     input_frames_cv2 = [cv2.resize(center_crop_cv2(pil_to_cv2(i)), imsize) for i in pils_from_video(driver_video)]
#     # input_frames_cv2 = [cv2.resize(pil_to_cv2(i), imsize) for i in pils_from_video(driver_video)]
#     lmk_extractor = LMKExtractor()

#     ref_frame =cv2.resize(cv2.imread(ref_image), (512, 512))
#     ref_det = lmk_extractor(ref_frame)

#     sequence_driver_det = []
#     try: 
#         for frame in input_frames_cv2:
#             result = lmk_extractor(frame)
#             assert result is not None, "{}, bad video, face not detected".format(driver_video)
#             sequence_driver_det.append(result)
#     except:
#         continue
#     print(len(sequence_driver_det))

#     # os.makedirs(save_dir, exist_ok=True)
#     # for i in range(len(sequence_driver_det)):
#     #     with open('{}/{}.pkl'.format(save_dir, i), 'wb') as file:
#     #         pickle.dump(sequence_driver_det[i]["lmks"] * imsize[0], file)



#         #[vis.draw_landmarks(imsize, i["lmks"], normed=True, white=True) for i in det_results]

#     pose_frames_driver = [vis.draw_landmarks((512, 512), i["lmks"], normed=True) for i in sequence_driver_det]
#     poses_add_driver = [(i * 0.5 + j * 0.5).clip(0,255).astype(np.uint8)  for i, j in zip(input_frames_cv2, pose_frames_driver)]

#     sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
#     for i in range(len(sequence_det_ms)):
#         tmp = {}
#         tmp["lmks"] = sequence_det_ms[i]
#         with open('{}_v1/{}.pkl'.format(save_dir, i), 'wb') as file:
#             pickle.dump(tmp, file)
#     pose_frames = [vis.draw_landmarks((512, 512), i, normed=False) for i in sequence_det_ms]
#     poses_add = [(i * 0.5 + ref_frame * 0.5).clip(0,255).astype(np.uint8) for i in pose_frames]

#     sequence_det_ms = motion_sync(sequence_driver_det, ref_det, per_landmark_align=False)
#     for i in range(len(sequence_det_ms)):
#         tmp = {}
#         tmp["lmks"] = sequence_det_ms[i]
#         with open('{}_v2/{}.pkl'.format(save_dir, i), 'wb') as file:
#             pickle.dump(tmp, file)
#     pose_frames_wo_lmkalign = [vis.draw_landmarks((512, 512), i, normed=False) for i in sequence_det_ms]
#     poses_add_wo_lmkalign = [(i * 0.5 + ref_frame * 0.5).clip(0,255).astype(np.uint8) for i in pose_frames_wo_lmkalign]

#     poses_cat = [np.concatenate([i, j, k], axis=1) for i, j, k in zip(poses_add_driver, poses_add_wo_lmkalign, poses_add)]

#     save_video_from_cv2_list(poses_cat, "./output/example2.mp4", fps=24.0)
#     # exit()
#     #embed()

#     #poses_cat = [(i * 0.5 + j * 0.5).clip(0,255).astype(np.uint8)  for i, j in zip(input_frames_cv2, pose_frames)]
#     #save_videos_from_pils([cv2_to_pil(i) for i in poses_cat], "./output/pose_cat.mp4", fps=24)