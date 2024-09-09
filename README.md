# MSc Capstone Project
- **team member: Pengxin WANG, Shenyang Tong, Jie Yan**

## File Structure
```
├── config/
├── data/
├── scripts/
│   ├── download_data.sh
│   ├── train.sh
│   └── test.sh    
├── weaver/
│   ├── __init__.py
│   ├── engines/
│   │   ├── train.py
│   │   └── test.py 
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── builder.py
│   │   ├── process.py
│   │   └── transform.py
│   ├── serialization/
│   ├── model/
│   ├── utils
│   │   ├── register.py
│   │   ├── logger.py
│   │   ├── structure.py
│   │   └── misc.py
│   ├── structure.py
│   └── pos_emb/
└──log/
```

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

## Visualization