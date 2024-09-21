# MSc Capstone Project
- **team member: Pengxin WANG, Shenyang Tong, Jie Yan**

## File Structure
```
├── config/
├── data/
├── scripts/
│   ├── download.sh
│   ├── train.sh
│   └── test.sh 
├── tools/
│   ├── vis.py
│   ├── train.py
│   └── test.py
└── weaver/   
    ├── __init__.py
    ├── engines/
    │   ├── hooks/
    │   │   ├── __init__.py
    │   │   ├── builder.py
    │   │   ├── default.py
    │   │   └── evaluator.py 
    │   ├── train.py
    │   ├── launch.py
    │   ├── defaults.py
    │   ├── train.py
    │   └── test.py 
    ├── datasets/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── builder.py
    │   ├── process.py
    │   └── transform.py
    ├── model/
    │   ├── losses/
    │   │   ├── __init__.py
    │   │   ├── builder.py
    │   │   ├── lovasz.py
    │   │   └── misc.py 
    │   ├── norm/
    │   │   └── pdnorm.py 
    │   ├── model_utils/
    │   │   ├── __init__.py
    │   │   ├── structure.py
    │   │   ├── misc.py
    │   │   └── serialization/
    │   │       ├── __init__.py
    │   │       ├── default.py
    │   │       ├── hilbert.py
    │   │       └── z_order.py 
    │   ├── __init__.py
    │   ├── builder.py
    │   ├── default.py
    │   ├── modules.py
    │   └── point_transformer_v3.py
    └── utils
        ├── register.py
        ├── logger.py
        ├── structure.py
        └── misc.py
```

## Preparation
- **connect to gpu:** `srun --gres=gpu:2 --time=02:00:00 --cpus-per-task=8 --pty --mail-type=ALL bash`

## Note:
- [A nice blog for registry in python](https://blog.csdn.net/weixin_44878336/article/details/133887655)
- do not set batch_size per cpu to 1, small bug will happen on cls_head

## Curent Task:
- [ ] Reproduce Point Transformer v3 on ModelNet40
    - current progress:
        - config: configs/ModelNet40/cls_ptv3_small.py
            - Val result: mIoU/mAcc/allAcc 0.7377/0.8217/0.8861
        - test time augmentation disabled 
        - [ ] why and how mIoU is evaluated in this task?
        - [ ] enable flash attention

## Data

### Data Example

### General Setting
- Point Module: A Dict(addict.Dict) storing necessary information of a batch of point cloud data 
    - coord: coordination of points, dtype=int32
    - offset: index to separate point clouds

- Note: for point cloud data, it's actually consisted of a batch of point cloud, using offset to indicate separation
    - points = [pc1, pc1, pc1, pc2, pc2]
    - offset = [3, 5]
    - classes = [chair, desk]

### Preprocessing
- Offset
- potential numerical issue(amp, automatic mixed precision, applied)
    - point.feat.dtype = torch.float16
    - point.coord.dtype = torch.float32
    - in flash attention, params are processed in half precision

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
- **mix_prob:**
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

### Attention Mechanism

### Patch Partition

### Patch Interaction

### Layer Normalization

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
### Shape Classification
- metrics: mAcc, allAcc

## Visualization