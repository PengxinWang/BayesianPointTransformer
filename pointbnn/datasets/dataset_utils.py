import random
from collections.abc import Mapping, Sequence
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(batch[0], Mapping)  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat([batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0)
    return batch

class CustomedDynamicDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, max_points_per_batch, mix_prob=0, drop_last=True, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.max_points_per_batch = max_points_per_batch
        self.mix_prob = mix_prob

    def point_collate_fn(self, batch):
        assert isinstance(batch[0], Mapping)  # currently, only support input_dict, rather than input_list
        batch = collate_fn(batch)
        if "offset" in batch.keys():
            # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
            if random.random() < self.mix_prob:
                batch["offset"] = torch.cat([batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0)
        return batch
    
    def __iter__(self):
        data_iter = super().__iter__()
        batch = []
        batch_points = 0
        self.end_epoch = False

        for idx, data_dict in enumerate(data_iter):
            if idx == len(data_iter)-1:
                self.end_epoch = True
            point_count = data_dict['coord'].shape[0]
            # If adding this sample exceeds the max points threshold, yield the batch
            if batch_points + point_count > self.max_points_per_batch:
                yield self.point_collate_fn(batch)
                batch = []
                batch_points = 0

            # Add sample to batch and update point count
            batch.append(data_dict)
            batch_points += point_count
        if batch and not self.drop_last:
            yield self.point_collate_fn(batch)

class CustomedDynamicDistributedSampler(DistributedSampler):
    def __init__(self,
                dataset: Dataset, 
                points_per_sample,
                num_replicas: int | None = None, 
                rank: int | None = None, 
                shuffle: bool = True, 
                seed: int = 0, 
                drop_last: bool = True) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.points_per_sample = points_per_sample
        self.indices = self.create_index_ranges()

    def create_index_ranges(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        total_points = sum(self.points_per_sample)
        max_points_per_device = int(total_points/self.num_replicas)
        points_per_sample = [self.points_per_sample[idx] for idx in indices]

        index_ranges = [[] for _ in range(self.num_replicas)]
        points_per_device = [0] * self.num_replicas
        device_id = 0
        for idx, num_point in zip(indices, points_per_sample):
            if device_id >= self.num_replicas:
                break
            if points_per_device[device_id] + num_point > max_points_per_device:
                device_id += 1
            else:
                index_ranges[device_id].append(idx)
                points_per_device[device_id] += num_point
        return index_ranges

    def __iter__(self):
        return iter(self.indices[self.rank])