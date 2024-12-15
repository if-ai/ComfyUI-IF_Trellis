<h1 align="center">ComfyUI-IF_Trellis</h1>
<p align="center"><a href="https://arxiv.org/abs/2412.01506"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://trellis3d.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/JeffreyXiang/TRELLIS'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>
ComfyUI TRELLIS is a large 3D asset generation in various formats, such as Radiance Fields, 3D Gaussians, and meshes. The cornerstone of TRELLIS is a unified Structured LATent (SLAT) representation that allows decoding to different output formats and Rectified Flow Transformers tailored for SLAT as the powerful backbones. 

![teaser (1)](https://github.com/user-attachments/assets/6eee56bd-0936-44a5-b843-be4e87be649f)

### Prerequisites
- **System**: The original code is currently tested only on **Linux**.  For windows setup, you may refer to [#3](https://github.com/microsoft/TRELLIS/issues/3)
- (This comfyui node is following this suggeted installation steps It works but you need to follow the steps as described in part one and two of the guide).
- Windows users need to use the Win_requirements.txt.
- Linux Users (Not tested Yet) use the linux_requirements.txt but I am still testing if it works with comfy in Linux I just added the original repo requirements.  
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A100 and A6000 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  This repo use **CUDA 12.4**.
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

  Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

### Installation Steps
1. Clone the repo:
    ```
    cd ComfyUI/Custom_nodes
    git clone --recurse-submodules https://github.com/if-ai/ComfyUI-IF_Trellis.git
    ```
## MUST HAVE `--recurse-submodules`

ONLY tested on windows but it should work easier in Linux without any issues or needing such specific stuffs.
***NOT tested or compatible with PORTABLE comfy embeded python env***
watch quick overview of setting the env if needed
[![Watch the video](https://img.youtube.com/vi/-vEpuYL9I3g/hqdefault.jpg)](https://www.youtube.com/watch?v=-vEpuYL9I3g)

You need to set up the environment first
follow this guide for the first part

Set the VSCode Cpp Envirronment as in the guide
  ```
  https://ko-fi.com/post/Installing-Triton-and-Sage-Attention-Flash-Attenti-P5P8175434
  ```
Setting up ComfyUI with the Xformers, flash attention, Sage-attention(Optional Recommended for Hunyuan and other Video models)


<!-- Installation -->
## 📦 Second part of the installation

Activate your comfy environment
```
(gen) PS D:\ComfyUI\custom_nodes\ComfyUI-IF_Trellis> micromamba activate gen
```
If you haven't set your vars or for some reason it can't compile some of this specially `nvdiffrast`
it doesn't hurt if you do it again now.
```
cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 "&&" powershell
```
You will see some message like this:

**********************************************************************
** Visual Studio 2019 Developer Command Prompt v16.11.41
** Copyright (c) 2021 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

```
pip install -r win_requirements.txt
```

```bash
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
New-Item -ItemType Directory -Force -Path C:\tmp\extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git C:\tmp\extensions\diffoctreerast
pip install C:\tmp\extensions\diffoctreerast
git clone https://github.com/autonomousvision/mip-splatting.git C:\tmp\extensions\mip-splatting
pip install C:\tmp\extensions\mip-splatting\submodules\diff-gaussian-rasterization\
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
git clone https://github.com/NVlabs/nvdiffrast.git C:\tmp\extensions\nvdiffrast
pip install C:\tmp\extensions\nvdiffrast
```


![blender_oWFuSjA6yq](https://github.com/user-attachments/assets/7d5fb61a-f2f6-4000-ab32-8555d5c6b7da)

![thorium_kTHxaiDIEV](https://github.com/user-attachments/assets/55b95ba4-13c2-4541-ab3a-4cc5750c6727)


## 🌟 Features
- **High Quality**: It produces diverse 3D assets at high quality with intricate shape and texture details.
- **Versatility**: It takes text or image prompts and can generate various final 3D representations including but not limited to *Radiance Fields*, *3D Gaussians*, and *meshes*, accommodating diverse downstream requirements.
- **Flexible Editing**: It allows for easy editings of generated 3D assets, such as generating variants of the same object or local editing of the 3D asset.

<!-- TODO List -->
## 🚧 TODO List
- [x] Release comfyUI-IF_Trellis
- [ ] 3D Viewport 
- [ ] OPT mode
- [ ] Fix BUGS
- [ ] Installation
