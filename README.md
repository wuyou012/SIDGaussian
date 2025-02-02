<h1 align="center"><strong>SIDGaussian</strong></h1>
<h2 align="center">See In Detail: Enhancing Sparse-view 3D Gaussian Splatting with Local Depth and Semantic Regularization</h2>

<p align="center">
  Zongqi He<sup>*</sup> ·
  <a href="http://zachary-zhexiao.github.io/">Zhe Xiao</a><sup>*</sup> ·
  Kin-Chung Chan ·
  <a href="https://yushenzuo.github.io/">Yushen Zuo</a><sup></sup> ·
  <a href="https://junxiao01.github.io/">Jun Xiao</a><sup>+</sup> ·
  <a href="https://www.eie.polyu.edu.hk/~enkmlam/">Kin-Man Lam</a><sup></sup>
</p>
<p align="center"><sup>*</sup>Equal Contribution · <sup>+</sup>Corresponding Author</p>

<h3 align="center">ICASSP 2025</h3>

<!-- <a align="center" href="https://arxiv.org/abs/2501.11508"><img src="https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red" alt="Paper"></a> -->

<!-- <center></center> -->

<!--[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2501.11508) -->

<div align="center">
    <a href="https://arxiv.org/abs/2501.11508">
        <img src="https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red" alt="Paper">
    </a>
</div>



<!-- <div style="display: flex; align - items: center; justify - content: center;">
    <h3>ICASSP 2025</h3>
    <a href="https://arxiv.org/abs/2501.11508"><img src="https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red" alt="Paper"></a>
</div> -->

<!-- [![Project Page](https://img.shields.io/badge/FSGS-Website-blue?logo=googlechrome&logoColor=blue)](https://zehaozhu.github.io/FSGS/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://youtu.be/CarJgsx3DQY)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVITA-Group%2FFSGS&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) -->


<!--
---------------------------------------------------
<p align="center" >
  <a href="">
    <img src="https://github.com/zhiwenfan/zhiwenfan.github.io/blob/master/Homepage_files/videos/FSGS_gif.gif?raw=true" alt="demo" width="85%">
  </a>
</p>
-->

## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate SIDGaussian
```
We use **CUDA 11.7** as our environment.

## Data Preparation
We use dense point cloud from [FSGS](https://github.com/VITA-Group/FSGS?tab=readme-ov-file#data-preparation) for initialization.

You may directly download through [this link](https://drive.google.com/drive/folders/1lYqZLuowc84Dg1cyb8ey3_Kb-wvPjDHA).
## Training
<!--
Train SIDGaussian on LLFF dataset with 3 views
``` 
for SCENE in fern flower fortress horns leaves orchids room trex
do
  CUDA_VISIBLE_DEVICES=0 python train.py --source_path dataset/nerf_llff_data/$SCENE --model_path output_llff/$SCENE --eval --n_views 3 --sample_pseudo_interval 1 --D 0.8 --W 0.5 --N 1
done
``` 


Train SIDGaussian on MipNeRF-360 dataset with 24 views
``` 
for SCENE in bonsai counter garden kitchen room stump bicycle
do
  CUDA_VISIBLE_DEVICES=0 python train.py --source_path /home/data1/mipnerf360/$SCENE --model_path output_mip/$SCENE --eval --n_views 24 --D 0.1 --W 0.25 --N 0.05
done
``` 
-->
Train SIDGaussian on LLFF dataset with 3 views
``` 
bash scripts_train/llff.sh
``` 

## Rendering
To render images:

```
python render.py --source_path dataset/nerf_llff_data/fern/  --model_path  output/fern --iteration 10000
```

To render a video:

```
python render.py --source_path dataset/nerf_llff_data/fern/  --model_path  output/fern --iteration 10000  --video  --fps 30
```

## Evaluation
The training code train.py automatically save evaluation scores, you can also run the following script to evaluate the model.

```
python metrics.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000
```

## Acknowledgement

Thanks to the following awesome open source projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [NeRF](https://github.com/bmild/nerf)
- [FSGS](https://github.com/VITA-Group/FSGS)

## Citation
If you find this project useful, please consider citing:
```
@article{he2025see,
  title={See In Detail: Enhancing Sparse-view 3D Gaussian Splatting with Local Depth and Semantic Regularization},
  author={He, Zongqi and Xiao, Zhe and Chan, Kin-Chung and Zuo, Yushen and Xiao, Jun and Lam, Kin-Man},
  journal={arXiv preprint arXiv:2501.11508},
  year={2025}
}
```

