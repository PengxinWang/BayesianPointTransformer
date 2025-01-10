# Master Capstone Project
Improve Point Transformer by Bayesian Perturbation for Uncertainty Quantification

## Uncertainty Quantification
- Reference: [What Uncertainty Do we need in Computer Vision?](https://arxiv.org/pdf/1703.04977)
<img src="imgs/uncertainty_vis.png" alt="UQ" title="UQ" width="300">
- Aleatoric(Data) vs. Epistemic(Model)



## Data
- [ ] ModelNet40
    - Shape Classification
- [ ] S3DIS
    - Indoor Semantic Segmentation
- [ ] ShapeNet
    - Part Segmentation
- how to download?
    - see scripts/download.sh

### Data Augmentation

## Model Structure

## Methods
### Rank-one Bayesian Perturbation
### Loss Function

## File Structure
**Note:** The framework, taking reference in mmDectection, is a little bit clumsy and complicated. The basic idea is decouple different modules, and 
```
├── config/                                                 
├── data/               
├── scripts/                                # Launcher
│   ├── ...
│   └── train.sh 
├── tools/                                  # Read Config
│   ├── ...
│   └── test.py
└── pointbnn/                               # Main Modules
    ├── engines/                            # Trainer, Tester, Hook
    ├── datasets/                           
    ├── model/                              
    └── utils/                              # misc
```

## Training Details
- **connect to gpu:** `srun --gres=gpu:2 --cpus-per-task=8 --pty --mail-type=ALL bash`
- 2 RTX 2080 Ti

## Experience records:
S3DIS:
exp0: ptv3(vanilla), ce, lovasz, rpe, patch_size=64, crop_n_points=102400
exp3: bnn, bce, lovasz, rpe, patch_size=64, sto_type=['heads', 'proj'], crop_n_points=102400
exp5: bnn, bce, lovasz, no rpe, patch_size=128, sto_type=['heads', 'proj'], crop_n_points=102400

ModelNet40:
exp0: ptv3(vanilla)
exp7: bnn, no rpe, patch_size=128

## Misc
- GEMM(General Matrix Multiplication)
    - [A nice blog](https://zhuanlan.zhihu.com/p/435908830)
    - core idea: partition

- [A nice pre on Point Cloud processing](https://www.youtube.com/watch?v=4gKYE9-YtP0)

- Learn more about sparse convolution: [Minkov Engine](https://github.com/NVIDIA/MinkowskiEngine)

- A nice paper to read: [superpoint graph clustering](https://arxiv.org/pdf/2401.06704)

- How to infuse 2D feature to 3D, and how does it help? [2D fusion](https://www.bilibili.com/read/cv33456793/)

- [PointNet BNN](https://github.com/biophase/PointNet-BNN)

- [Some people work on bayesian deep learning](https://www.x-mol.com/paper/1788682254484697088/t)

- [A paper repository for Point Cloud Understanding](https://github.com/Yochengliu/awesome-point-cloud-analysis)

## The project structure is based on PointCept, Torch-Uncertainty

