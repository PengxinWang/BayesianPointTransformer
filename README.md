# MSc Capstone Project
- **team member: Pengxin WANG, Shenyang Tong, Jie Yan**

## Preparation
- **connect to gpu:** `srun --gres=gpu:2 --time=02:00:00 --cpus-per-task=8 --pty --mail-type=ALL bash`

## Curent Task:
- [ ] Reproduce Point Transformer v3 on ModelNet40 shape classification
    - current progress:
        - config: configs/ModelNet40/cls_ptv3_small.py
            - val result: mIoU/mAcc/allAcc 0.7377/0.8217/0.8861
        - test time augmentation disabled 
    - TODO
        - [ ] enable flash attention
        - [ ] enable and test pdnorm

- [ ] Reproduce PTv3 on S3DIS semantic segmentation
    - current progress:
    - TODO
        - [ ] enable and test DiceLoss fine tuning

- [ ] Visualize serialization
    - what to use? Open3D? Unity?

## Data

### General Setting
- Point Module: A Dict(addict.Dict) storing necessary information of a batch of point cloud data 
    - coord: coordination of points, dtype=int32
    - offset: index to separate point clouds

- **Offset**: for point cloud data, it's actually consisted of a batch of point cloud, using offset to indicate separation
    - points = [pc1, pc1, pc1, pc2, pc2]
    - offset = [3, 5]
    - classes = [chair, desk]

### Preprocessing
- Offset
- potential numerical issue(amp, automatic mixed precision, applied)
    - point.feat.dtype = torch.float16
    - point.coord.dtype = torch.float32
    - in flash attention, params are processed in half precision
- for evaluation

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
    - raw scale: [-1,1]
    - processed: .pth

### Data Augmentation
- what is point cloud jittering?
- what does anisotropic mean in random_scale()?
- mixed3D

### Post Processing
- Random_scale for 10 times for ensembling(TTA)

## Config
### base config: default_runtime.py
- **enable_amp=False:** automatic mixed precision
- **sync_bn:** synchronizing batch normalization for multi GPU training
- **empty_cache:** empty GPU cache, exchange time for space
- **find_unused_parameters:**
- **mix_prob:** related to Mixed3D data augmentation
- **param_dicts:** allow lr scale to certain param groups

## Model Structure
- **Model size**
    - (cls_ptv3_base.py) n_params: about 40M
    - (cls_ptv3_small.py) n_params: about 9M(9792296)
    
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