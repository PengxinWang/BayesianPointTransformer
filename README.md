# MSc Capstone Project
- **team member: Pengxin WANG, Shenyang Tong, Jie Yan**

## Preparation
- **connect to gpu:** `srun --gres=gpu:2 --time=02:00:00 --cpus-per-task=8 --pty --mail-type=ALL bash`

## Curent Task:
- [x] Reproduce Point Transformer v3 on ModelNet40 shape classification
- [x] Reproduce PTv3 on S3DIS semantic segmentation
- [x] Finish code for PT-BNN
    - [ ] projection only; attentino only
- [x] Visualize serialization
    - [ ] clustering property; density imbalance
- [ ] Visualize uncertainty quantification
- [x] Dynamic DataLoading, try to make total points in a batch same
- [x] Balanced CE Loss
- [ ] Centralized RPE(cRPE)
- [ ] Grouped vector attention
- [ ] Tune proper grid_size and sphere crop

## Note:
- Train grid size need to match val/test grid size

## Data

### General Setting
- Point Module: A Dict(addict.Dict) storing necessary information of a batch of point cloud data 
    - offset: index to separate point clouds

### Preprocessing

### ModelNet40
- how many classes/shape are there?
    - 40
- how many point clouds are there in the dataset?
    - trainset: 9843
    - val/testset: 2468
- how many points are sampled in each point cloud?
    - raw: 10000 points
    - processed: 
- what is the format of raw data?
    - raw: .txt
    - processed: .pth
- scale of points
    - raw scale: [-1,1]
    - GridSample: gird_size=0.05

### S3DIS
- download
    - fill out the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)
    - [Paper](https://ieeexplore.ieee.org/document/7780539)
- data



### Data Augmentation
- what is point cloud jittering?
- what does anisotropic mean in random_scale()?
- mixed3D

### Post Processing
- Random_scale for 10 times for ensembling(TTA)

## Model Structure
- **Model size**
    - (cls_ptv3_base.py) n_params: about 40M
    - (cls_ptv3_small.py) n_params: about 9M(9792296)
    - (semseg_ptv3_small.py) n_params: about 10M(10379109)
    
## Methods
### Serialization
- Hilbert encoding and decoding algorithm

### Positional Embedding
- **RPE(Relative Positional Embedding):**

- **SubMConv3D(Submanimold Sparse Convolution):** 
    - [Original Paper](https://arxiv.org/pdf/1711.10275)
    - [Official Document](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md)
    - input_feat.shape = output_feat.shape

### Attention Mechanism

### Patch Partition

### Patch Interaction

### Layer Normalization
- Layer Norm is more suitable for data with variable length

### Loss Function

### Optimization
- Optimizer: SGD, Adam, AdamW
    - AdamW

- Param Group: speicify different learning hparam setting(lr, weight_decay, etc.) to different param groups
    - **lr:** default=0.001, "block"=0.0001 
    - **weight_decay:** default=0.01

- Scheduler: OneCycleLR
- Scaler: for calibrating precision between different GPUs
- train_epochs: 300(default), 6(pilot)

## Evaluation

**Note:** In shape classification, each shape is an instance; in semantic segmentation, each point/pixel is an instance.

### Shape Classification
- **Note:** 'Acc' here is actually recall, calculated by TP/TP+FN; IoU is calculated by TP/TP+FP+FN
- metrics: mAcc, allAcc, mIoU

## Visualization

## File Structure
```
├── config/
├── data/
├── scripts/                                
│   ├── ...
│   └── train.sh 
├── tools/                                 
│   ├── ...
│   └── test.py
└── weaver/   
    ├── __init__.py
    ├── engines/                           # hooks, trainers, testers, ddp laucher...
    │   ├── hooks/
    │   │   ├── ...
    │   │   └── evaluator.py 
    │   ├── ...
    │   └── test.py 
    ├── datasets/                          # dataset, transform...
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── ...
    │   └── transform.py
    ├── model/
    │   ├── losses/
    │   ├── norm/                          # pdnorm
    │   ├── model_utils/                    
    │   │   ├── structure.py               # Point data dict
    │   │   ├── ...
    │   │   └── serialization/             # serialization
    │   ├── ...
    │   └── point_transformer_v3.py
    └── utils/                             # registration, ddp, logger, timer, optimizer...
        ├── register.py
        ├── logger.py
        ├── ...
        └── misc.py
```

## Misc
- GEMM(General Matrix Multiplication)
    - [A nice blog](https://zhuanlan.zhihu.com/p/435908830)
    - core idea: partition

- [A nice pre on Point Cloud processing](https://www.youtube.com/watch?v=4gKYE9-YtP0)

- A possible improvement for sparse convolution: [Minkov Engine](https://github.com/NVIDIA/MinkowskiEngine)

- do not set batch_size per cpu to 1, small bug will happen on cls_head

- A new paper to read: [superpoint graph clustering](https://arxiv.org/pdf/2401.06704)

- How to infuse 2D feature to 3D, and how does it help? [2D fusion](https://www.bilibili.com/read/cv33456793/)

- [PointNet BNN](https://github.com/biophase/PointNet-BNN)

- [Some people work on bayesian deep learning](https://www.x-mol.com/paper/1788682254484697088/t)

- [A paper repository for Point Cloud Understanding](https://github.com/Yochengliu/awesome-point-cloud-analysis)

## The project structure is based on PointCept, Torch-Uncertainty

