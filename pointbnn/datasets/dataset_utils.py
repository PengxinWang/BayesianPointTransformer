import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


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
    def __init__(self, dataset, max_points_per_batch, mix_prob=0, *args, **kwargs):
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
        for data_dict in data_iter:
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