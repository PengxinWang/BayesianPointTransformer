import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

def pc_normalize(pc):
    """
    normalize the point cloud with center 0 and max norm 1
    pc.shape = [num_point, dim_point]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    norm_max = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / norm_max
    return pc

def farthest_point_sample(point, n_sample:int):
    """
    sample n_sample of centroids from a set of point
    point.shape = [N, D]
    return: point.shape = [n_sample, D]
    Note: sample in place
    """
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros(n_sample)
    distance = np.ones(N)*1e10 # the distance of each point to its nearest centroid
    farthest_point = np.random.randint(0, N) # randomly initialze the index of a farthest point
    for i in range(n_sample):
        centroids[i] = farthest_point
        centroid_xyz = xyz[farthest_point, :]
        dist = np.sum((xyz - centroid_xyz)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest_point = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataset(Dataset):
    def __init__(self, 
                 root, 
                 num_point=1024, 
                 n_cat=40, 
                 uniform_sample=False, 
                 use_normals=False, 
                 split='train', 
                 process_data=False):
        super().__init__()
        self.root = root
        self.num_point = num_point
        self.process_data = process_data
        self.uniform_sample = uniform_sample
        self.use_normals = use_normals
        self.n_cat = n_cat
        self.split = split

        self.catfile = os.path.join(self.root, f'modelnet{self.n_cat}_shape_names.txt')
        self.catlist = [line.rstrip() for line in open(self.catfile)]
        self.cat_to_id = dict(zip(self.catlist, range(len(self.cat))))
        
        pc_ids = [line.rstrip() for line in open(os.path.join(self.root, f'modelnet{self.n_cat}_{self.split}.txt'))]
        self.pc_cats = ['_'.join(x.split('_')[0:-1]) for x in pc_ids]
        self.pc_paths = [os.path.join(self.root, self.pc_cats[i], pc_ids[i], f'.txt') for i in range(len(pc_ids))]
        print(f'The size of {self.split} data is {len(self.pc_paths)}')

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, index):
        pc_path = self.pc_paths[index]
        pc = np.loadtxt(pc_path, delimiter=',').astype(np.float32)
        if self.uniform_sample:
            N = pc.shape[0]
            pc_indexes = np.random.choice(N, self.num_point, replace=False)
            pc = pc[pc_indexes, :]
        else:
            pc = farthest_point_sample(pc, self.num_point)
        pc[:, :3] = pc_normalize(pc[:, :3])
        if not self.use_normals:
            pc = pc[:, :3]
        cat = self.pc_cats[index]
        id = np.int32(self.cat_to_id[cat])
        return pc, id