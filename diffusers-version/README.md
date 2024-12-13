# Tora diffusers-version

This path contains the diffusers-version of Tora. It is independent from the original Tora code which based on SwissArmyTransformer.

## Installation

Please make sure your Python version is between 3.10 and 3.12, inclusive of both 3.10 and 3.12.

```bash
# Install Pytorch (we use Pytorch 2.4.0) and torchvision following the official instructions: https://pytorch.org/get-started/previous-versions/. For example:
conda create -n tora python==3.10
conda activate tora
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements
cd diffusers-version
pip install -r requirements.txt
```

## Model Weights

- #### HuggingFace

```bash
# This can be faster
pip install "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Le0jc/Tora_T2V_diffusers --local-dir ckpts/Tora_T2V_diffusers
```

or

```bash
# use git
git lfs install
git clone https://huggingface.co/Le0jc/Tora_T2V_diffusers
```

- #### ModelScope

```bash
pip install modelscope
modelscope download --model Alibaba_Research_Intelligence_Computing/Tora_T2V_diffusers --local_dir ckpts/Tora_T2V_diffusers
```

or

```bash
git lfs install
git clone https://www.modelscope.cn/Alibaba_Research_Intelligence_Computing/Tora_T2V_diffusers.git
```

## Inference

```bash
cd diffusers-version
python inference.py --prompt "A squirrel gathering nuts." --model_path ckpts/Tora_T2V_diffusers --output_path ./output.mp4 --generate_type t2v --point_path ../sat/trajs/pause.txt --enable_model_cpu_offload --enable_slicing --enable_tiling --enable_sageattention --enable_compile
```

- If your VRAM is still not enough, you can replace "--enable_model_cpu_offload" to "--enable_sequential_cpu_offload" and try again. This can reduce the VRAM usage to about 5 GiB. Note that sequential_cpu_offload is much slower.
- If you have enough VRAM, you can disable cpu offload, VAE slicing and tiling, to speed up the inference.
- Note that --enable_compile will speed up inference at the cost of slowing down the first inference step.