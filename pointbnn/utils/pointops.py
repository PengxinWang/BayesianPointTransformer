import torch
import torch.nn as nn
import numpy as np

def knn_query(k, coords, offsets, origin_coords, origin_offsets):
    """
    coords: torch.tensor()
    """
    assert k==1, "current only support nearest point for interpolation"
    res = []
    if isinstance(origin_offsets, np.ndarray):
        print(type(origin_offsets))
        origin_offsets=torch.from_numpy(origin_offsets)
    if offsets[0] != 0:
        offsets = nn.functional.pad(offsets, (1, 0))
    if origin_offsets[0] != 0:
        origin_offsets = nn.functional.pad(origin_offsets, (1, 0))

    for i in range(len(offsets)-1):
        bs, be = offsets[i], offsets[i+1]
        bs_ori, be_ori = origin_offsets[i], origin_offsets[i+1]
        batch_coords = coords[bs:be]
        batch_coords_ori = origin_coords[bs_ori:be_ori]
        distances = torch.norm(batch_coords.unsqueeze(1) - batch_coords_ori.unsqueeze(0), dim=2)
        cloest_indices = torch.argmin(distances, dim=0)
        global_indices = bs + cloest_indices
        res.append(global_indices)
    res = torch.cat(res, dim=0)
    return res
