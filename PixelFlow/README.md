<div align="center">

<h1> PixelFlow: Pixel-Space Generative Models with Flow </h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.07963-b31b1b.svg)](https://arxiv.org/abs/2504.07963)&nbsp;
[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-blue)](https://huggingface.co/spaces/ShoufaChen/PixelFlow)&nbsp;


![pixelflow](https://github.com/user-attachments/assets/7e2e4db9-4b41-46ca-8d43-92f2b642a676)

</div>




> [**PixelFlow: Pixel-Space Generative Models with Flow**](https://arxiv.org/abs/2504.07963)<br>
> [Shoufa Chen](https://www.shoufachen.com), [Chongjian Ge](https://chongjiange.github.io/), [Shilong Zhang](https://jshilong.github.io/), [Peize Sun](https://peizesun.github.io/), [Ping Luo](http://luoping.me/)
> <br>The University of Hong Kong, Adobe<br>

## Introduction
We present PixelFlow, a family of image generation models that operate directly in the raw pixel space, in contrast to the predominant latent-space models. This approach simplifies the image generation process by eliminating the need for a pre-trained Variational Autoencoder (VAE) and enabling the whole model end-to-end trainable. Through efficient cascade flow modeling, PixelFlow achieves affordable computation cost in pixel space. It achieves an FID of 1.98 on 256x256 ImageNet class-conditional image generation benchmark. The qualitative text-to-image results demonstrate that PixelFlow excels in image quality, artistry, and semantic control. We hope this new paradigm will inspire and open up new opportunities for next-generation visual generation models.


## Model Zoo

| Model     | Task           | Params | FID  | Checkpoint |
|:---------:|:--------------:|:------:|:----:|:----------:|
| PixelFlow | class-to-image | 677M  | 1.98 | [🤗](https://huggingface.co/ShoufaChen/PixelFlow-Class2Image) |
| PixelFlow | text-to-image  | 882M  | N/A  | [🤗](https://huggingface.co/ShoufaChen/PixelFlow-Text2Image)  |


## Setup

### 1. Create Environment
```bash
conda create -n pixelflow python=3.12
conda activate pixelflow
```
### 2. Install Dependencies:
* [PyTorch 2.6.0](https://pytorch.org/) — install it according to your system configuration (CUDA version, etc.).
* [flash-attention v2.7.4.post1](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1): optional, required only for training.
* Other packages: `pip3 install -r requirements.txt`


## Demo [![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-blue)](https://huggingface.co/spaces/ShoufaChen/PixelFlow)


We provide an online [Gradio demo](https://huggingface.co/spaces/ShoufaChen/PixelFlow) for class-to-image generation. 

You can also easily deploy both class-to-image and text-to-image demos locally by:

```bash
python app.py --checkpoint /path/to/checkpoint --class_cond  # for class-to-image
```
or
```bash
python app.py --checkpoint /path/to/checkpoint  # for text-to-image
```


## Training

### 1. ImageNet Preparation

- Download the ImageNet dataset from [http://www.image-net.org/](http://www.image-net.org/).
- Use the [extract_ILSVRC.sh]([extract_ILSVRC.sh](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)) to extract and organize the training and validation images into labeled subfolders.

### 2. Training Command

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py configs/pixelflow_xl_c2i.yaml
```

## Evaluation (FID, Inception Score, etc.)

We provide a [sample_ddp.py](sample_ddp.py) script, adapted from [DiT](https://github.com/facebookresearch/DiT), for generating sample images and saving them both as a folder and as a .npz file. The .npz file is compatible with ADM's TensorFlow evaluation suite, allowing direct computation of FID, Inception Score, and other metrics.


```bash
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --pretrained /path/to/checkpoint
```


## BibTeX
```bibtex
@article{chen2025pixelflow,
  title={PixelFlow: Pixel-Space Generative Models with Flow},
  author={Chen, Shoufa and Ge, Chongjian and Zhang, Shilong and Sun, Peize and Luo, Ping},
  journal={arXiv preprint arXiv:2504.07963},
  year={2025}
}
```
