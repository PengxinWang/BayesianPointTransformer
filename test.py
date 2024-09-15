import torch.nn as nn
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        return
name = DefaultClassifier.__name__
print(f'type: {type(name)}')
print(name)

import torch
print(torch.cuda.device_count())