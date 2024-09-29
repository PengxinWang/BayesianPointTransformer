import torch
import torch.nn as nn
import torch_scatter

from weaver.models.losses import build_criteria
from weaver.models.model_utils.structure import Point
from weaver.models.model_utils.bayesian import StoLinear, StoLayer
from .builder import MODELS, build_model
# from weaver.utils.logger import get_root_logger
# debug_logger = get_root_logger(log_file=f"/userhome/cs2/pxwang24/capstone/Weaver/exp/S3DIS/semseg_ptbnn_small/train.log", file_mode='w')

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        feat = point.feat
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class BayesSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        n_components=4,
        n_samples=4,
        backbone=None,
        criteria=None,
        kl_weight=1.0,
        entropy_weight=1.0,
        stochastic=True,
        prior_mean=1.0, 
        prior_std=0.40, 
        post_mean_init=(1.0, 0.05), 
        post_std_init=(0.25, 0.10),
    ):
        super().__init__()
        self.n_components = n_components
        self.n_samples = n_samples
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.stochastic=stochastic
        self.prior_mean = prior_mean 
        self.prior_std = prior_std 
        self.post_mean_init = post_mean_init 
        self.post_std_init = post_std_init
        self.seg_head = (StoLinear(in_features=backbone_out_channels, 
                                   out_features=num_classes, 
                                   n_components=n_components,
                                   prior_mean=prior_mean, 
                                   prior_std=prior_std, 
                                   post_mean_init=post_mean_init, 
                                   post_std_init=post_std_init))
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.sto_layers = [m for m in self.modules() if isinstance(m, (StoLayer))]
    
    def kl_and_entropy(self):
        kl = sum([m._kl() for m in self.sto_layers])
        entropy = sum([m._entropy() for m in self.sto_layers])
        return (kl, entropy)

    def forward(self, input_dict):
        point = Point(input_dict)
        point.get_samples(self.n_samples)
        point = self.backbone(point)
        feat = point["feat"]
        seg_logits = self.seg_head(feat)

        # train
        if self.training:
            nll = self.criteria(seg_logits, point["segment"])
            kl, entropy = self.kl_and_entropy()
            kl = kl - self.entropy_weight * entropy
            return dict(nll=nll, kl=kl)
        # eval
        elif "segment" in input_dict.keys():
            seg_logits = seg_logits.view(-1, self.n_components, seg_logits.size(1))
            seg_logits = torch.mean(seg_logits, dim=1)
            nll = self.criteria(seg_logits, input_dict["segment"])
            kl, entropy = self.kl_and_entropy()
            kl = kl - self.entropy_weight * entropy
            return dict(nll=nll, kl=kl, seg_logits=seg_logits)
        # test
        else:
            seg_logits = seg_logits.view(-1, self.n_components, seg_logits.size(1))
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
                )
        feat = point.feat
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)