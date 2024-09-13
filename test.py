from weaver.utils.optimizer import build_optimizer
from addict import Dict
import torch

test_cfg = Dict()
test_cfg['params']=[torch.tensor(0)]
test_cfg['type']='Adam'
class model():
    def __init__(self) -> None:
        model._parameters = [torch.tensor(1)]
    
    def parameters(self):
        return model._parameters

test_model = model()

build_optimizer(cfg=test_cfg, model=test_model)

