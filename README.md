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
- **connect to gpu:** `srun --gres=gpu:1 --time=02:00:00 --cpus-per-task=4 --pty --mail-type=ALL bash`

## Note:
- [A nice blog for registry in python](https://blog.csdn.net/weixin_44878336/article/details/133887655)

## Curent Task:
- [ ] Reproduce Point Transformer v3 on ModelNet40

## Data

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

### ModelNet40
- how many classes/shape are there?
    - 40
- how many point clouds are there in the dataset?
    - trainset: 9843
    - val/testset: 2468
- how many points are sampled in each point cloud?
    - after processing: 10000 points
- what is the format of raw data?
    - raw: .txt
    - processed: .pth

### Data Augmentation
- what is point cloud jittering?
- why there are 6 consecutive random_scale() applied during test?
- what does anisotropic mean in random_scale()?

## Config
### base config: default_runtime.py
- **enable_amp=False:** automatic mixed precision
- **sync_bn:** synchronizing batch normalization for multi GPU training
- **empty_cache:** empty GPU cache, exchange time for space
- **find_unused_parameters:**
- **mix_prob:**
- **param_dicts:** allow lr scale to certain param groups

## Methods
### Serialization
#### Hilbert encoding and decoding algorithm

### Positional Embedding
- **RPE(Relative Positional Embedding):**

### Attention Mechanism

### Patch Partition

### Patch Interaction

### Layer Normalization

### Loss Function

### Optimization
- Optimizer: AdamW
- Scheduler: OneCycleLR
- Scaler: for calibrating precision between different GPUs

## Visualization