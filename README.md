<div align="center">

<img src="sat/assets/icon/icon.jpg" width="250"/>

<h2><center>[🔥CVPR'25]Tora: Trajectory-oriented Diffusion Transformer for Video Generation</h2>

Zhenghao Zhang\*, Junchao Liao\*, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, Weizhi Wang

\* equal contribution
<br>

<a href='https://arxiv.org/abs/2407.21705'><img src='https://img.shields.io/badge/ArXiv-2407.21705-red'></a>
<a href='https://ali-videoai.github.io/tora_video/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a>
<a href="https://github.com/alibaba/Tora"><img src='https://img.shields.io/badge/Github-Link-orange'></a>
<a href='https://www.modelscope.cn/studios/xiaoche/Tora'><img src='https://img.shields.io/badge/🤖_ModelScope-ZH_demo-%23654dfc'></a>
<a href='https://www.modelscope.cn/studios/Alibaba_Research_Intelligence_Computing/Tora_En'><img src='https://img.shields.io/badge/🤖_ModelScope-EN_demo-%23654dfc'></a>
<br>

<a href='https://modelscope.cn/models/xiaoche/Tora'><img src='https://img.shields.io/badge/🤖_ModelScope-T2V/I2V_weights(SAT)-%23654dfc'></a>
<a href='https://modelscope.cn/models/Alibaba_Research_Intelligence_Computing/Tora_T2V_diffusers'><img src='https://img.shields.io/badge/🤖_ModelScope-T2V_weights(diffusers)-%23654dfc'></a>
<br>

<a href='https://huggingface.co/Alibaba-Research-Intelligence-Computing/Tora'><img src='https://img.shields.io/badge/🤗_HuggingFace-T2V/I2V_weights(SAT)-%23ff9e0e'></a>
<a href='https://huggingface.co/Alibaba-Research-Intelligence-Computing/Tora_T2V_diffusers'><img src='https://img.shields.io/badge/🤗_HuggingFace-T2V_weights(diffusers)-%23ff9e0e'></a>
</div>

This is the official repository for paper "Tora: Trajectory-oriented Diffusion Transformer for Video Generation".

## 💡 Abstract

Recent advancements in Diffusion Transformer (DiT) have demonstrated remarkable proficiency in producing high-quality video content. Nonetheless, the potential of transformer-based diffusion models for effectively generating videos with controllable motion remains an area of limited exploration. This paper introduces Tora, the first trajectory-oriented DiT framework that integrates textual, visual, and trajectory conditions concurrently for video generation. Specifically, Tora consists of a Trajectory Extractor (TE), a Spatial-Temporal DiT, and a Motion-guidance Fuser (MGF). The TE encodes arbitrary trajectories into hierarchical spacetime motion patches with a 3D video compression network. The MGF integrates the motion patches into the DiT blocks to generate consistent videos following trajectories. Our design aligns seamlessly with DiT’s scalability, allowing precise control of video content’s dynamics with diverse durations, aspect ratios, and resolutions. Extensive experiments demonstrate Tora’s excellence in achieving high motion fidelity, while also meticulously simulating the movement of physical world.

## 📣 Updates
- `2025/07/08` 🔥🔥 Our latest work, [Tora2](https://ali-videoai.github.io/Tora2_page/), has been accepted by ACM MM25. Tora2 builds on Tora with design improvements, enabling enhanced appearance and motion customization for multiple entities.
- `2025/05/24` We open-sourced a LoRA-finetuned model of [Wan](https://github.com/Wan-Video/Wan2.1). It turns things in the image into fluffy toys. Check this out: https://github.com/alibaba/wan-toy-transform
- `2025/01/06` 🔥🔥We released Tora Image-to-Video, including inference code and model weights.
- `2024/12/13` SageAttention2 and model compilation are supported in diffusers version. Tested on the A10, these approaches speed up every inference step by approximately 52%, except for the first step.
- `2024/12/09` 🔥🔥Diffusers version of Tora and the corresponding model weights are released. Inference VRAM requirements are reduced to around 5 GiB. Please refer to [this](diffusers-version/README.md) for details.
- `2024/11/25` 🔥Text-to-Video training code released.
- `2024/10/31` Model weights uploaded to [HuggingFace](https://huggingface.co/Le0jc/Tora). We also provided an English demo on [ModelScope](https://www.modelscope.cn/studios/Alibaba_Research_Intelligence_Computing/Tora_En).
- `2024/10/23` 🔥🔥Our [ModelScope Demo](https://www.modelscope.cn/studios/xiaoche/Tora) is launched. Welcome to try it out! We also upload the model weights to [ModelScope](https://www.modelscope.cn/models/xiaoche/Tora).
- `2024/10/21` Thanks to [@kijai](https://github.com/kijai) for supporting Tora in ComfyUI! [Link](https://github.com/kijai/ComfyUI-CogVideoXWrapper)
- `2024/10/15` 🔥🔥We released our inference code and model weights. **Please note that this is a CogVideoX version of Tora, built on the CogVideoX-5B model. This version of Tora is meant for academic research purposes only. Due to our commercial plans, we will not be open-sourcing the complete version of Tora at this time.**
- `2024/08/27` We released our v2 paper including appendix.
- `2024/07/31` We submitted our paper on arXiv and released our project page.

## 📑 Table of Contents

- [🎞️ Showcases](#%EF%B8%8F-showcases)
- [✅ TODO List](#-todo-list)
- [🧨 Diffusers verision](#-diffusers-verision)
- [🐍 Installation](#-installation)
- [📦 Model Weights](#-model-weights)
- [🔄 Inference](#-inference)
- [🖥️ Gradio Demo](#%EF%B8%8F-gradio-demo)
- [🧠 Training](#-training)
- [🎯 Troubleshooting](#-troubleshooting)
- [🤝 Acknowledgements](#-acknowledgements)
- [📄 Our previous work](#-our-previous-work)
- [📚 Citation](#-citation)

## 🎞️ Showcases

https://github.com/user-attachments/assets/949d5e99-18c9-49d6-b669-9003ccd44bf1

https://github.com/user-attachments/assets/7e7dbe87-a8ba-4710-afd0-9ef528ec329b

https://github.com/user-attachments/assets/4026c23d-229d-45d7-b5be-6f3eb9e4fd50

All videos are available in this [Link](https://cloudbook-public-daily.oss-cn-hangzhou.aliyuncs.com/Tora_t2v/showcases.zip)

## ✅ TODO List

- [x] Release our inference code and model weights
- [x] Provide a ModelScope Demo
- [x] Release our training code
- [x] Release diffusers version and optimize the GPU memory usage
- [x] Release complete version of Tora

## 🧨 Diffusers verision

Please refer to [the diffusers version](diffusers-version/README.md) for details.

## 🐍 Installation

Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.

```bash
# Clone this repository.
git clone https://github.com/alibaba/Tora.git
cd Tora

# Install Pytorch (we use Pytorch 2.4.0) and torchvision following the official instructions: https://pytorch.org/get-started/previous-versions/. For example:
conda create -n tora python==3.10
conda activate tora
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
cd modules/SwissArmyTransformer
pip install -e .
cd ../../sat
pip install -r requirements.txt
cd ..
```

## 📦 Model Weights

### Folder Structure

```
Tora
└── sat
    └── ckpts
        ├── t5-v1_1-xxl
        │   ├── model-00001-of-00002.safetensors
        │   └── ...
        ├── vae
        │   └── 3d-vae.pt
        ├── tora
        │   ├── i2v
        │   │   └── mp_rank_00_model_states.pt
        │   └── t2v
        │       └── mp_rank_00_model_states.pt
        └── CogVideoX-5b-sat # for training stage 1
            └── mp_rank_00_model_states.pt
```

### Download Links

*Note: Downloading the `tora` weights requires following the [CogVideoX License](CogVideoX_LICENSE).* You can choose one of the following options: HuggingFace, ModelScope, or native links.
After downloading the model weights, you can put them in the `Tora/sat/ckpts` folder.

#### HuggingFace

```bash
# This can be faster
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Alibaba-Research-Intelligence-Computing/Tora --local-dir ckpts
```

or

```bash
# use git
git lfs install
git clone https://huggingface.co/Alibaba-Research-Intelligence-Computing/Tora
```

#### ModelScope

- SDK

```bash
from modelscope import snapshot_download
model_dir = snapshot_download('xiaoche/Tora')
```

- Git

```bash
git clone https://www.modelscope.cn/xiaoche/Tora.git
```

#### Native

- Download the VAE and T5 model following [CogVideo](https://github.com/THUDM/CogVideo/blob/main/sat/README.md#2-download-model-weights):
    - VAE: https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
    - T5: [text_encoder](https://huggingface.co/THUDM/CogVideoX-2b/tree/main/text_encoder), [tokenizer](https://huggingface.co/THUDM/CogVideoX-2b/tree/main/tokenizer)
- Tora t2v model weights: [Link](https://cloudbook-public-daily.oss-cn-hangzhou.aliyuncs.com/Tora_t2v/mp_rank_00_model_states.pt). Downloading this weight requires following the [CogVideoX License](CogVideoX_LICENSE).

## 🔄 Inference

### Text to Video
It requires around 30 GiB GPU memory tested on NVIDIA A100.

```bash
cd sat
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=$N_GPU sample_video.py --base configs/tora/model/cogvideox_5b_tora.yaml configs/tora/inference_sparse.yaml --load ckpts/tora/t2v --output-dir samples --point_path trajs/coaster.txt --input-file assets/text/t2v/examples.txt
```

You can change the `--input-file` and `--point_path` to your own prompts and trajectory points files. Please note that the trajectory is drawn on a 256x256 canvas.

Replace `$N_GPU` with the number of GPUs you want to use.

### Image to Video

```bash
cd sat
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=$N_GPU sample_video.py --base configs/tora/model/cogvideox_5b_tora_i2v.yaml configs/tora/inference_sparse.yaml --load ckpts/tora/i2v --output-dir samples --point_path trajs/sawtooth.txt --input-file assets/text/i2v/examples.txt --img_dir assets/images --image2video
```

The first frame images should be placed in the `--img_dir`. The names of these images should be specified in the corresponding text prompt in `--input-file`, seperated by `@@`.

### Recommendations for Text Prompts

For text prompts, we highly recommend using GPT-4 to enhance the details. Simple prompts may negatively impact both visual quality and motion control effectiveness.

You can refer to the following resources for guidance:

- [CogVideoX Documentation](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py)
- [OpenSora Scripts](https://github.com/hpcaitech/Open-Sora/blob/main/scripts/inference.py)

## 🖥️ Gradio Demo

Usage:

```bash
cd sat
python app.py --load ckpts/tora/t2v
```

## 🧠 Training

### Data Preparation

Following this guide https://github.com/THUDM/CogVideo/blob/main/sat/README.md#preparing-the-dataset, structure the datasets as follows:

```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```

Training data examples are in `sat/training_examples`

### Text to Video

It requires around 60 GiB GPU memory tested on NVIDIA A100.

Replace `$N_GPU` with the number of GPUs you want to use.

- Stage 1

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=$N_GPU train_video.py --base configs/tora/model/cogvideox_5b_tora.yaml configs/tora/train_dense.yaml --experiment-name "t2v-stage1"
```

- Stage 2

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=$N_GPU train_video.py --base configs/tora/model/cogvideox_5b_tora.yaml configs/tora/train_sparse.yaml --experiment-name "t2v-stage2"
```

## 🎯 Troubleshooting

### 1. ValueError: Non-consecutive added token...

Upgrade the transformers package to 4.44.2. See [this](https://github.com/THUDM/CogVideo/issues/213) issue.

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects that have been instrumental in the development of our project:

- [CogVideo](https://github.com/THUDM/CogVideo): An open source video generation framework by THUKEG.
- [Open-Sora](https://github.com/hpcaitech/Open-Sora): An open source video generation framework by HPC-AI Tech.
- [MotionCtrl](https://github.com/TencentARC/MotionCtrl): A video generation model supporting motion control by ARC Lab, Tencent PCG.
- [ComfyUI-DragNUWA](https://github.com/chaojie/ComfyUI-DragNUWA): An implementation of DragNUWA for ComfyUI.

Special thanks to the contributors of these libraries for their hard work and dedication!

## 📄 Our previous work

- [AnimateAnything: Fine Grained Open Domain Image Animation with Motion Guidance](https://github.com/alibaba/animate-anything)

## 📚 Citation

```bibtex
@inproceedings{zhang2025tora,
  title={Tora: Trajectory-oriented diffusion transformer for video generation},
  author={Zhang, Zhenghao and Liao, Junchao and Li, Menghao and Dai, Zuozhuo and Qiu, Bingxue and Zhu, Siyu and Qin, Long and Wang, Weizhi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2063--2073},
  year={2025}
}
```
