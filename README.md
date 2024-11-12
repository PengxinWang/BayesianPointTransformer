# MSc Capstone Project
- **team member: Pengxin WANG, Shenyang Tong, Jie Yan**

## Preparation
- **connect to gpu:** `srun --gres=gpu:2 --cpus-per-task=8 --pty --mail-type=ALL bash`

## Experience records:
S3DIS:
exp0: ptv3(vanilla), ce, lovasz, rpe
exp1: bnn, bce, lovasz, rpe, kl_reweighing, sto_type = ['head']
exp2: bnn, bce, no_norm
exp3: bnn, bce, ln_norm
exp4: bnn, bce, bn_norm

ModelNet40:
[x] exp0: ptv3(vanilla)
[x] exp1: sto_type = ['head']
[x] exp2: sto_type = ['atten']
[x] exp3: sto_type = ['head', 'atten']
exp4: sto_type = ['proj']
exp5: sto_type = ['cpe']
exp6: sto_type = ['atten', 'proj', 'cpe']

exp7: ptv3(no dropOut, no dropPath)
exp8: ptv3(dropPath)
exp9: ptv3(dropOut head)
exp10: ptv3(dropOut qkv proj)
exp11: ptv3(dropOut proj)
## Data

### ShapeNet
- download

  link: https://pan.baidu.com/s/1wRA7zPytCCBx9b_jDjiVEg?pwd=wl08 
  password: wl08 

### Data Augmentation

### Post Processing

## Model Structure
- **Model size**
    - (cls_ptv3_base.py) n_params: about 40M
    - (cls_ptv3_small.py) n_params: about 9M(9792296)
    - (semseg_ptv3_small.py) n_params: about 10M(10379109)
    
## Methods
### Serialization

### Positional Embedding
- **RPE(Relative Positional Embedding):**

- **SubMConv3D(Submanimold Sparse Convolution):** 
    - [Original Paper](https://arxiv.org/pdf/1711.10275)
    - [Official Document](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md)
    - input_feat.shape = output_feat.shape

### Loss Function

## Evaluation

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

