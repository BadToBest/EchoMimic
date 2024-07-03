<h1 align='center'>EchoMimic: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning</h1>

<div align='center'>
    <a href='https://github.com/' target='_blank'>Zhiyuan Chen</a><sup>*</sup>&emsp;
    <a href='https://github.com/' target='_blank'>Jiajiong Cao</a><sup>*</sup>&emsp;
    <a href='https://github.com/' target='_blank'>Zhiquan Chen</a><sup></sup>&emsp;
    <a href='https://github.com/' target='_blank'>Yuming Li</a><sup></sup>&emsp;
    <a href='https://github.com/' target='_blank'>Chenguang Ma</a><sup></sup>
</div>
<div align='center'>
    *Equal Contribution.
</div>

<div align='center'>
Terminal Technology Department, Alipay, Ant Group.
</div>

<div align='center'>
    <a href=''><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='assets/wechat.jpeg'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
</div>

## Gallery 

### Audio Driven (Chinese)

[//]: # (<table class="center">)

[//]: # ()
[//]: # (<tr>)

[//]: # (    <td width=20% style="border: none">)

[//]: # (        <video controls autoplay loop src="./contents/demo_videos/chinese/ch_01.mp4" muted="false"></video>)

[//]: # (    </td>)

[//]: # (    <td width=20% style="border: none">)

[//]: # (        <video controls autoplay loop src="./contents/demo_videos/chinese/ch_02.mp4" muted="false"></video>)

[//]: # (    </td>)

[//]: # (    <td width=20% style="border: none">)

[//]: # (        <video controls autoplay loop src="./contents/demo_videos/chinese/ch_03.mp4" muted="false"></video>)

[//]: # (    </td>)

[//]: # (    <td width=20% style="border: none">)

[//]: # (        <video controls autoplay loop src="./contents/demo_videos/chinese/ch_04.mp4" muted="false"></video>)

[//]: # (    </td>)

[//]: # (    <td width=20% style="border: none">)

[//]: # (        <video controls autoplay loop src="./contents/demo_videos/chinese/ch_05.mp4" muted="false"></video>)

[//]: # (    </td>)

[//]: # (</tr>)

[//]: # ()
[//]: # (</table>)

### Audio Driven (English)


### Audio Driven (Sing)

### Landmark Driven

### Audio + Selected Landmark Driven


## Release Plans

|  Status  | Milestone                                                                |    ETA     |
|:--------:|:-------------------------------------------------------------------------|:----------:|
|    🚀    | Inference source code meet everyone on GitHub                            | TBD |
|    🚀    | Pretrained models trained on English and Mandarin Chinese to be released | TBD |
|     🚀     | Pretrained models with better pose control to be released                | TBD |
|     🚀     | Pretrained models with better sing performance to be released            | TBD |
|    🚀    | Large-Scale and High-resolution Chinese-Based Talking Head Dataset       |    TBD     |




## Installation

### Build Environtment

We Recommend a python version `>=3.10` and cuda version `=11.7`. Then build environment as follows:

```shell
# [Optional] Create a virtual env
python -m venv .venv
source .venv/bin/activate
# Install with pip:
pip install -r requirements.txt
```

### Download weights

**Automatically downloading**: You can run the following command to download weights automatically:

```shell
python tools/download_weights.py
```

Weights will be placed under the `./pretrained_weights` direcotry. The whole downloading process may take a long time.

**Manually downloading**: You can also download weights manually, which has some steps:

1. Download our trained [weights](https://huggingface.co/patrolli/AnimateAnyone/tree/main), which include four parts: `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth` and `motion_module.pth`.

2. Download pretrained weight of based models and other components: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

3. Download dwpose weights (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`) following [this](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet).

Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

Note: If you have installed some of the pretrained models, such as `StableDiffusion V1.5`, you can specify their paths in the config file (e.g. `./config/prompts/animation.yaml`).

## Training and Inference 

### Inference

Here is the cli command for running inference scripts:

```shell
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 64
```

You can refer the format of `animation.yaml` to add your own reference images or pose videos. To convert the raw video into a pose video (keypoint sequence), you can run with the following command:

```shell
python tools/vid2pose.py --video_path /path/to/your/video.mp4
```

### <span id="train"> Training </span>
todo

## Large Scale Chinese-based Talking Head Dataset

## Disclaimer

The codes and dataset are intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## Acknowledgements

We first thank the authors of [AnimateAnyone](). Additionally, we would like to thank the contributors to the [majic-animate](https://github.com/magic-research/magic-animate), [animatediff](https://github.com/guoyww/AnimateDiff) and [Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone) repositories, for their open research and exploration. Furthermore, our repo incorporates some codes from [dwpose](https://github.com/IDEA-Research/DWPose) and [animatediff-cli-prompt-travel](https://github.com/s9roll7/animatediff-cli-prompt-travel/), and we extend our thanks to them as well.
