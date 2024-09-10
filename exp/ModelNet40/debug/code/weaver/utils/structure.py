import torch
from addict import Dict

from weaver.serialization.default import encode, decode
from .utils import offset2batch, batch2offset

class Point(Dict):
    """
    A dictionary-like structure that contains the 3D point cloud data and additional attributes.

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);

    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization
        >>> Args: 
            order: str or list(str)
            supported_orders: {'z', 'z-trans', 'hilbert', 'hilbert-trans'}
        >>> Return:
            len(order)*n_points
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()

        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        # depth*3 bits are used to describe position, while the rest are used to descrite batch/offset index
        assert depth * 3 + len(self.offset).bit_length() <= 63
        assert depth <= 16

        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

if __name__ == '__main__':
    test_points = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [0.5, 0.6, 0.7], [0.5, .5, .5], [.6, .7, .4]])
    point = Point(
        coord=test_points,  
        grid_size=0.01,  
        offset=torch.tensor([2,4,5]),
    )

    orders = ['z', 'hilbert', 'z-trans', 'hilbert-trans']
    # Perform serialization
    point.serialization(order=orders)

    # Print out the result
    print("Serialized Code:", point["serialized_code"])
    print("Serialized Order:", point["serialized_order"])
    print("Serialized Inverse:", point["serialized_inverse"])