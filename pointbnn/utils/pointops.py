import numpy as np

def knn_query(coords, offsets, origin_coords, origin_offsets, k=1):
    assert k==1, "current only support nearest point for interpolation"
    res = []
    if offsets[0] != 0:
        offsets = np.insert(offsets, 0, 0)
    if origin_offsets[0] != 0:
        origin_offsets = np.insert(origin_offsets, 0, 0)

    for i in range(len(offsets)-1):
        bs, be = offsets[i], offsets[i+1]
        bs_ori, be_ori = origin_offsets[i], origin_offsets[i+1]
        batch_coords = coords[bs:be]
        batch_coords_ori = origin_coords[bs_ori:be_ori]
        distances = np.linalg.norm(batch_coords[:,np.newaxis,:] - batch_coords_ori[np.newaxis,:,:], axis=2)
        cloest_indices = np.argmin(distances, axis=0)
        global_indices = bs + cloest_indices
        res.append(global_indices)
    res = np.concatenate(res, axis=0)
    return res
