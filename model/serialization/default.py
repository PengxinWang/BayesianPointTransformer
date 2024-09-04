import torch
from model.serialization.hilbert import encode as hilbert_encode_
from model.serialization.hilbert import decode as hilbert_decode_

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        raise NotImplementedError
        # code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        raise NotImplementedError
        # code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 2 | code
    return code

@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        pass
        # grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch

def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=2, num_bits=depth)

def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=2, num_bits=depth)

if __name__=='__main__':
    import math
    n = 30  # number of points
    dim = 224  # dimensions of the image
    num_bits = int(math.log2(dim)) + 1

    # Generate random points in a 32x32 grid
    points = torch.randint(0, dim, (n, 2))
    hilbert_code = encode(grid_coord=points, depth=num_bits, order='hilbert')
    print(hilbert_code)
    # points_decoded = decode(hilbert_code, depth=num_bits, order='hilbert')
    # print(points_decoded==points)