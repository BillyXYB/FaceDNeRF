# FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models (NeurIPS 2023)
![Picture1](https://github.com/BillyXYB/FDNeRF/assets/49705209/70f8c597-f9db-482c-b585-28709ff6f468)


## Abstract
The ability to create high-quality 3D faces from a single image has become in-
creasingly important with wide applications in video conferencing, AR/VR, and
advanced video editing in movie industries. In this paper, we propose Face Diffu-
sion NeRF(FDNeRF), a new generative method to reconstruct high-quality Face
NeRFs from single images, complete with semantic editing and relighting capabili-
ties. FDNeRF utilizes high-resolution 3D GAN inversion and expertly trained 2D
latent-diffusion model, allowing users to manipulate and construct Face NeRFs in
zero-shot learning without the need for explicit 3D data. With carefully designed
illumination and identity preserving loss, as well as multi-modal pre-training, FD-
NeRF offers users unparalleled control over the editing process enabling them
to create and edit face NeRFs using just single-view images, text prompts, and
explicit target lighting. The advanced features of FDNeRF have been designed to
produce more impressive results than existing 2D editing approaches that rely on
2D segmentation maps for editable attributes. Experiments show that our FDNeRF
achieves exceptionally realistic results and unprecedented flexibility in editing
compared with state-of-the-art 3D face reconstruction and editing methods

![pipeline](https://github.com/BillyXYB/FDNeRF/assets/49705209/5af0e7fe-806b-4f9a-a68b-092917753496)


## Text-Conditioned 3D Editing on Single Image (include other domians)
https://github.com/BillyXYB/FDNeRF/assets/49705209/a6e9a410-2487-41d5-ab77-23fc6992dff2





## Explicit View-consistant 3D Relighting

https://github.com/BillyXYB/FDNeRF/assets/49705209/e1c3fcef-7fe3-432f-b58c-60aa0a798dcb



## Text-Condition Generation (include other domians)

https://github.com/BillyXYB/FDNeRF/assets/49705209/caf123ce-a63a-4692-beed-be685d3b29a9

## Requirements

* We recommend Linux for performance and compatibility reasons.
* 1&ndash;2 high-end NVIDIA GPUs. We have done all testing and development using V100s, RTX3090s and RTX4090s.
* 64-bit Python 3.9, cuda11.3, and PyTorch 1.11.0 (or later). See https://pytorch.org for PyTorch install instructions.
* Since we use the EG3D as our backbone, Please see **[eg3d](https://github.com/NVlabs/eg3d)** official repo for EG3D installation. Or directly install following conda environment.
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Anaconda3 to create and activate your Python environment:
  - `cd FaceDNeRF`
  - `conda env create -f environment.yml`
  - `conda activate facednerf`

## Data preparation

We use the same camera pose convention as [eg3d](https://github.com/NVlabs/eg3d), please refer to this [script](https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/ffhq/preprocess_in_the_wild.py) that can preprocess in-the-wild images compatible with our camera pose convention.

We also provide test data in the `./test_data`

## Download pre-trained models
1. VGG16 pre-trained model: you can download vgg16.pt from https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt and save it to `./network`.
2. Since EG3D is the backbone of our model, please download the ffhqrebalanced512-64.pkl by the following command and place it under `./network`.
```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/eg3d/1/files?redirect=true&path=ffhqrebalanced512-128.pkl' -O ffhqrebalanced512-128.pkl
```
3. The ArcFace facial recognition network is used in the ID loss. The weights can be downloaded from [here](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing).

## Image editing
The backnone of this implementation is [eg3d](https://github.com/NVlabs/eg3d). For detailed instructions, please refer to the comments in the  `./script.py`

```
python script.py
```

Results will be saved to `./output`. If you encounte the error: `Function 'PowBackward0' returned nan values in its 0th output.` caused by the `File "XXX/anaconda3/envs/facednerf/lib/python3.9/site-packages/kornia/color/rgb.py", line 199, in rgb_to_linear_rgb`, please replace the line 199 code: `lin_rgb: torch.Tensor = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)` with `lin_rgb: torch.Tensor = torch.where(image > 0.04045, torch.pow(((image.abs() + 0.055) / 1.055), 2.4), image / 12.92)`.Since the `.abs()` function is added to ensure that the base number is positive, the torch.power function becomes differentiable, and the gradients can be passed through it.

## Relight images
we have provided some illumination functions `getLightIntensity`in the file `./editors/w_plus_editor.py`. If you want to customize you own illumination function, pleace also name it `getLightIntensity` and replace the original one.

## Paper & Citation
Link to [**Paper**](https://arxiv.org/abs/2306.00783) 

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhang2024facednerf,
  title={FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models},
  author={Zhang, Hao and DAI, Tianyuan and Xu, Yanbo and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
