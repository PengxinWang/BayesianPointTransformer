import torch

@torch.inference_mode()
def offset2bincount(offset):
    """
    count bins(number of points in each group)
    offset: a 1d array of integer, which indicates the ending index of each group
    e.g. offset = [3, 6, 9]
         return: [3, 3, 3]
    """
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    """
    convert offset index into batch index
    e.g. offset = [3, 6, 9]
         return: [0, 0, 0, 1, 1, 1, 2, 2, 2]
    """
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    """
    convert batch index into offset index
    e.g. batch = [0, 0, 1, 1]
         return: [2, 4]
    """
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    """ 
    return a flattened view of the off-diagonal elements of a square matrix
    e.g. x = [[1,2,2],[2,1,2],[2,2,1]]
         off_diagonal(x) = [2,2,2,2,2,2]
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()