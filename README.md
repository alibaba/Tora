<div align="center">

<img src="sat/assets/icon/icon.jpg" width="250"/>

<h2><center>Tora: Trajectory-oriented Diffusion Transformer for Video Generation</h2>

Zhenghao Zhang\*, Junchao Liao\*, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, Weizhi Wang

\* equal contribution

<a href='https://arxiv.org/abs/2407.21705'><img src='https://img.shields.io/badge/ArXiv-2407.21705-red'></a>
<a href='https://ali-videoai.github.io/tora_video/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> ![views](https://visitor-badge.laobi.icu/badge?page_id=alibaba.Tora&left_color=gray&right_color=green)
<a href="git clone https://github.com/alibaba/Tora/stargazers"><img src="https://img.shields.io/github/stars/alibaba/Tora?style=social"></a>

</div>

This is the official repository for paper "Tora: Trajectory-oriented Diffusion Transformer for Video Generation".

## 💡 Abstract

Recent advancements in Diffusion Transformer (DiT) have demonstrated remarkable proficiency in producing high-quality video content. Nonetheless, the potential of transformer-based diffusion models for effectively generating videos with controllable motion remains an area of limited exploration. This paper introduces Tora, the first trajectory-oriented DiT framework that integrates textual, visual, and trajectory conditions concurrently for video generation. Specifically, Tora consists of a Trajectory Extractor (TE), a Spatial-Temporal DiT, and a Motion-guidance Fuser (MGF). The TE encodes arbitrary trajectories into hierarchical spacetime motion patches with a 3D video compression network. The MGF integrates the motion patches into the DiT blocks to generate consistent videos following trajectories. Our design aligns seamlessly with DiT’s scalability, allowing precise control of video content’s dynamics with diverse durations, aspect ratios, and resolutions. Extensive experiments demonstrate Tora’s excellence in achieving high motion fidelity, while also meticulously simulating the movement of physical world.

## 📣 Updates

- `2024/10/15` 🔥🔥We released our inference code and model weights. **Please note that this is a CogVideoX version of Tora, built on the CogVideoX-5B model. This version of Tora is meant for academic research purposes only. Due to our commercial plans, we will not be open-sourcing the complete version of Tora at this time.**
- `2024/08/27` We released our v2 paper including appendix.
- `2024/07/31` We submitted our paper on arXiv and released our project page.

## 📑 Table of Contents

- [Showcases](#🎞️-showcases)
- [TODO List](#✅-todo-list)
- [Installation](#🐍-installation)
- [Model Weights](#📦-model-weights)
- [Inference](#🔄-inference)
- [Gradio Demo](#🖥️-gradio-demo)
- [Acknowledgements](#🤝-acknowledgements)
- [Our previous work](#📄-our-previous-work)
- [Citation](#📚-citation)

## 🎞️ Showcases

All videos are available in this [Link](https://cloudbook-public-daily.oss-cn-hangzhou.aliyuncs.com/Tora_t2v/showcases.zip)

## ✅ TODO List

- [x] Release our inference code and model weights
- [ ] Provide a ModelScope Demo
- [ ] Release our training code
- [ ] Release complete version of Tora

## 🐍 Installation

```bash
# Clone this repository.
git clone https://github.com/alibaba/Tora.git
cd tora

# Install Pytorch (we use Pytorch 2.4.0) and torchvision following the official instructions: https://pytorch.org/get-started/previous-versions/. For example:
conda create -n tora python==3.10
conda activate tora
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
cd modules/SwissArmyTransformer
pip install -e .
cd ../../sat
pip install -r requirements.txt
```

## 📦 Model Weights

### Folder Structure

```
sat
└── ckpts
    ├── t5-v1_1-xxl
    ├── vae
    │   └── 3d-vae.pt
    └── tora
        └── t2v
            └── mp_rank_00_model_states.pt
```

### Download Links

- Download the VAE and T5 model following [CogVideo](https://github.com/THUDM/CogVideo/blob/main/sat/README.md#2-download-model-weights)
- Tora t2v model weights: [Link](https://cloudbook-public-daily.oss-cn-hangzhou.aliyuncs.com/Tora_t2v/mp_rank_00_model_states.pt)

## 🔄 Inference

It requires around 30 GiB GPU memory tested on NVIDIA A100.

```bash
cd sat
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=$N_GPU sample_video.py --base configs/tora/model/cogvideox_5b_tora.yaml configs/tora/inference_sparse.yaml --load ckpts/tora/t2v --output-dir samples --point_path trajs/coaster.txt --input-file assets/text/t2v/examples.txt
```

You can change the `--input-file` and `--point_path` to your own prompts and trajectory points files. Replace `$N_GPU` with the number of GPUs you want to use.

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
@misc{zhang2024toratrajectoryorienteddiffusiontransformer,
      title={Tora: Trajectory-oriented Diffusion Transformer for Video Generation},
      author={Zhenghao Zhang and Junchao Liao and Menghao Li and Zuozhuo Dai and Bingxue Qiu and Siyu Zhu and Long Qin and Weizhi Wang},
      year={2024},
      eprint={2407.21705},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.21705},
}
```
