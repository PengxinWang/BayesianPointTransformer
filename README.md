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
    ├── serialization/
    ├── model/
    └── utils
        ├── register.py
        ├── logger.py
        ├── structure.py
        └── misc.py
```

## Preparation
- **connect to gpu:** `srun --gres=gpu:2 --time=04:00:00 --cpus-per-task=4 --pty --mail-type=ALL bash`

## Note:
- [A nice blog for registry in python](https://blog.csdn.net/weixin_44878336/article/details/133887655)

## Curent Task:
- [ ] Reproduce Point Transformer v3 on ModelNet40

## Data
- Point Module: A Dict(addict.Dict) storing necessary information of a batch of point cloud data 
    - coord: coordination of points, dtype=int32
    - offset: index to separate point clouds

- question(wpx): What is the default setting of grid size? What is the scale of point coordinate?

- Note: for point cloud data, it's actually consisted of a batch of point cloud, using offset to indicate separation
    - points = [pc1, pc1, pc1, pc2, pc2]
    - offset = [3, 5]
    - classes = [chair, desk]

## Methods
### Serialization
#### Hilbert encoding and decoding algorithm

### Positional Embedding

### Attention Mechanism

### Patch Partition

### Patch Interaction

### Layer Normalization

### Loss Function

### Optimization
- Optimizer: AdamW
- Scheduler: OneCycleLR

## Visualization