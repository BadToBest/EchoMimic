from PIL import Image
import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
import pathlib
import os
from IPython import embed

def pil_to_cv2(pil):
    return cv2.cvtColor(np.array(pil).astype(np.uint8), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB).astype(np.uint8))

def center_crop_cv2(cv2_pic):
    h, w = cv2_pic.shape[0], cv2_pic.shape[1]
    if h > w:
        return cv2_pic[(h - w) // 2 : (h - w) // 2 + w, :]
    else:
        return cv2_pic[:, (w - h) // 2 : (w - h) // 2 + h]

def pils_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pils = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(center_crop_cv2(frame), (512, 512))
        pils.append(cv2_to_pil(frame))

    return pils

def save_videos_from_pils(pils, path, fps=24):
    width, height = pils[0].size
    print(width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    

    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)

    output_name = pathlib.Path(path).stem
    temp_output_path = path.replace(output_name, output_name + '-temp')
    videowrite = cv2.VideoWriter(temp_output_path, fourcc, fps, (height, width))
    for pil in pils:
        frame = pil_to_cv2(pil)
        print(frame.shape, frame.min(), frame.max())
        videowrite.write(frame)

    videowrite.release()
    """
    embed()

    cmd = (f'{get_ffmpeg_exe()} -i "{temp_output_path}"'
           f'-map 0:v -map 1:a -c:v h264 -shortest -y "{path}" -loglevel quiet')
    os.system(cmd)
    os.remove(temp_output_path)
    """
def save_video_from_cv2_list(pic_cv2_list, output_path, fps=30.0):
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    num_frames = len(pic_cv2_list)
    height, width = pic_cv2_list[0].shape[:2]
    
    #video_tensor = video_tensor[0, ...]
    #_, num_frames, height, width = video_tensor.shape

    output_name = pathlib.Path(output_path).stem
    temp_output_path = output_path.replace(output_name, output_name)
    video_writer = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(num_frames):
        #frame_tensor = video_tensor[:, i, ...]  # [c, h, w]
        #frame_tensor = frame_tensor.permute(1, 2, 0)  # [h, w, c]

        #frame_image = (frame_tensor * 255).numpy().astype(np.uint8)
        #frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
        frame_image = pic_cv2_list[i].astype(np.uint8)
        video_writer.write(frame_image)

    video_writer.release()


#ffmpeg -i input_file -c:v libx264 -crf 20 -c:a aac -strict experimental -b:a 192k output_file