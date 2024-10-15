<p align="center">
    <img src="assets/logo2.png"/>
</p>
<p align="center">
<b>Generate vivid Images for <i>Any</i> (Chinese / English) text</b>
</p>

CogView2 is a hierarchical transformer (6B-9B-9B parameters) for text-to-image generation in general domain. This implementation is based on the [SwissArmyTransformer](https://github.com/THUDM/SwissArmyTransformer) library (v0.2).

* **Read** our paper [CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers](https://arxiv.org/pdf/2105.13290.pdf) on ArXiv for a formal introduction. The *LoPAR* accelarate the generation and *CogLM* enables the model for bidirectional completion.
* **Run** our pretrained models from text-to-image generation or text-guided completion! Please use A100 GPU.
* **Cite** our paper if you find our work is helpful~ 
```
@article{ding2022cogview2,
  title={CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers},
  author={Ding, Ming and Zheng, Wendi and Hong, Wenyi and Tang, Jie},
  journal={arXiv preprint arXiv:2204.14217},
  year={2022}
}
```

## Getting Started
### Setup
* Hardware: Linux servers with Nvidia A100s are recommended, but it is also okay to run the pretrained models with smaller `--max-inference-batch-size` or training smaller models on less powerful GPUs.
* Environment: install dependencies via `pip install -r requirements.txt`. 
* LocalAttention: Make sure you have CUDA installed and compile the local attention kernel.
```shell
git clone https://github.com/Sleepychord/Image-Local-Attention
cd Image-Local-Attention && python setup.py install
```
If you don't install this kernel, you can also run the first stage (20*20 tokens) via `--only-first-stage` for text-to-image generation.

### Download
Our code will automatically download or detect the models into the path defined by envrionment variable `SAT_HOME`. You can download from [here](https://model.baai.ac.cn/model-detail/100041) and place them (folders named `coglm`/`dsr`/`itersr`) under `SAT_HOME`. 

### Text-to-Image Generation
```
./script/text2image.sh --input-source input.txt
```
Arguments useful in inference are mainly:
* `--input-source [path or "interactive"]`. The path of the input file, can also be "interactive", which will launch a CLI.
* `--output-path [path]`. The folder containing the results.
* `--batch-size [int]`. The number of samples will be generated per query.
* `--max-inference-batch-size [int]`. Maximum batch size per forward. Reduce it if OOM. 
* `--debug`. Only save concatenated images for all generated samples, and name them by input text and date. 
* `--with-id`. When it toggled, you must specify an "id" before each input, e.g. `001\t一个漂亮的女孩`, \t denoting TAB (**NOT space**). It will generate `batch-size` split images in a folder named "id" for each input. Confict with `--debug`.
* `--device [int]`. Running on which GPU. 
* `--inverse-prompt`. Use the perplexity to generate the original text to sort the generated images.
* `--only-first-stage`. 
* `--style`. The style of the generated images, choices=['none', 'mainbody', 'photo', 'flat', 'comics', 'oil', 'sketch', 'isometric', 'chinese', 'watercolor']. The default style is `mainbody`, usually an isolated object with white background.

You'd better specify a environment variable `SAT_HOME` to specify the path to store the downloaded model.

Chinese input is usually much better than English input.

### Text-guided Completion
```
./script/text_guided_completion.sh --input-source input_comp.txt
```
The format of input is `text	image_path	h0	w0	h1	w1`, where all the separation are **TAB** (**NOT space**). The image at `image_path` will be center-cropped to `480*480` pixels and mask the square from `(h0,w0)`to `(h1,w1)`. These coordinations are range from 0 to 1. The model will fill the square with object described in `text`. Please use a square much **larger than the desired region**.  

## Gallery
![more_samples](https://github.com/THUDM/CogView2/files/8553662/big.1.pdf)
