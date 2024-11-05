import torch
import torch.nn as nn
import torch_scatter
import math

from pointbnn.models.losses import build_criteria
from pointbnn.models.model_utils.structure import Point
from pointbnn.models.model_utils.bayesian import StoLinear, StoLayer
from pointbnn.utils.misc import point_wise_entropy
from .builder import MODELS, build_model

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        stochastic=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.stochastic = stochastic

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
        n_training_samples=1,
        backbone=None,
        criteria=None,
        kl_weight_init = 1e-2,
        kl_weight_final=1.0,
        entropy_weight=1.0,
        stochastic=True,
        prior_mean=1.0, 
        prior_std=0.40, 
        post_mean_init=(1.0, 0.05), 
        post_std_init=(0.40, 0.20),
        stochastic_modules = ['atten', 'proj', 'cpe', 'head']
    ):
        super().__init__()
        self.n_classes = num_classes
        self.n_components = n_components
        self.n_training_samples = n_training_samples
        self.n_samples = n_samples
        self.kl_weight_init = kl_weight_init
        self.kl_weight_final = kl_weight_final
        self.entropy_weight = entropy_weight
        self.stochastic = stochastic
        self.prior_mean = prior_mean 
        self.prior_std = prior_std 
        self.post_mean_init = post_mean_init 
        self.post_std_init = post_std_init
        if 'head' in stochastic_modules:
            self.seg_head = (StoLinear(in_features=backbone_out_channels, 
                                   out_features=num_classes, 
                                   n_components=n_components,
                                   prior_mean=prior_mean, 
                                   prior_std=prior_std, 
                                   post_mean_init=post_mean_init, 
                                   post_std_init=post_std_init))
        else:
            self.seg_head = (nn.Linear(in_features=backbone_out_channels,
                                    out_features=num_classes))

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.sto_layers = [m for m in self.modules() if isinstance(m, (StoLayer))]
    
    def kl_and_entropy(self):
        kl = torch.mean(torch.stack([m._kl() for m in self.sto_layers]))
        entropy = torch.mean(torch.stack([m._entropy() for m in self.sto_layers]))
        return (kl, entropy)

    def forward(self, input_dict):
        point = Point(input_dict)
        if self.training:
            point.get_samples(self.n_training_samples)
        else:
            point.get_samples(self.n_samples)
        point = self.backbone(point)
        feat = point["feat"]
        seg_logits = self.seg_head(feat)

        # train
        if self.training:
            target_segments = point["segment"] 
            nll = self.criteria(seg_logits, target_segments)
            kl, entropy = self.kl_and_entropy()
            kl = kl - self.entropy_weight * entropy
            # kl = kl * self.n_training_samples
            kl = kl*self.n_training_samples/math.sqrt(target_segments.shape[0]+1)
            # kl = kl*self.n_training_samples/target_segments.shape[0]
            return dict(nll=nll, kl=kl)
        # eval
        elif "segment" in input_dict.keys():
            seg_logits = seg_logits.view(-1, self.n_samples, seg_logits.size(1))
            mean_seg_logits = torch.mean(seg_logits, dim=1)
            nll = self.criteria(mean_seg_logits, input_dict["segment"])
            kl, entropy = self.kl_and_entropy()
            kl = kl - self.entropy_weight * entropy
            return dict(nll=nll, kl=kl, seg_logits=mean_seg_logits)
        # test
        else:
            seg_logits = seg_logits.view(-1, self.n_samples, seg_logits.size(1))
            mean_seg_logits = torch.mean(seg_logits, dim=1)
            predictive = point_wise_entropy(seg_logits, type='predictive')
            aleatoric = point_wise_entropy(seg_logits, type='aleatoric')
            epistemic = predictive - aleatoric
            return dict(seg_logits=mean_seg_logits, 
                        seg_logits_samples=seg_logits,
                        epistemic=epistemic, 
                        aleatoric=aleatoric)

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
        stochastic=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.stochastic = stochastic
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

@MODELS.register_module()
class BayesClassifier(nn.Module):
    def __init__(
        self,
        num_classes=40,
        backbone_embed_dim=256,
        n_components=4,
        n_samples=4,
        n_training_samples=1,
        backbone=None,
        criteria=None,
        kl_weight_init = 1e-2,
        kl_weight_final=1.0,
        entropy_weight=1.0,
        stochastic=True,
        prior_mean=1.0, 
        prior_std=0.40, 
        post_mean_init=(1.0, 0.05), 
        post_std_init=(0.40, 0.20),
        stochastic_modules = ['atten', 'proj', 'cpe', 'head']
    ):
        super().__init__()
        self.n_classes = num_classes
        self.n_components = n_components
        self.n_training_samples = n_training_samples
        self.n_samples = n_samples
        self.kl_weight_init = kl_weight_init
        self.kl_weight_final = kl_weight_final
        self.entropy_weight = entropy_weight
        self.stochastic = stochastic
        self.prior_mean = prior_mean 
        self.prior_std = prior_std 
        self.post_mean_init = post_mean_init 
        self.post_std_init = post_std_init
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        if 'head' in stochastic_modules:
            self.cls_head = nn.Sequential(
                StoLinear(backbone_embed_dim, 256, 
                          n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                          post_mean_init=post_mean_init, post_std_init=post_std_init),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                StoLinear(256, 128,
                          n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                          post_mean_init=post_mean_init, post_std_init=post_std_init),
                nn.ReLU(inplace=True),
                StoLinear(128, num_classes,
                          n_components=n_components, prior_mean=prior_mean, prior_std=prior_std,
                          post_mean_init=post_mean_init, post_std_init=post_std_init),
            )
        else:
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
        self.sto_layers = [m for m in self.modules() if isinstance(m, (StoLayer))]

    def kl_and_entropy(self):
        kl = torch.mean(torch.stack([m._kl() for m in self.sto_layers]))
        entropy = torch.mean(torch.stack([m._entropy() for m in self.sto_layers]))
        return (kl, entropy)

    def forward(self, input_dict):
        point = Point(input_dict)
        if self.training:
            point.get_samples(self.n_training_samples)
        else:
            point.get_samples(self.n_samples)
        point = self.backbone(point)
        point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point["offset"], (1, 0)),
                reduce="mean",
                )
        feat = point.feat
        cls_logits = self.cls_head(feat)
        if self.training:
            target_cats = input_dict["category"]
            nll = self.criteria(cls_logits, target_cats)
            kl, entropy = self.kl_and_entropy()
            kl = kl - self.entropy_weight * entropy
            # kl = kl * self.n_training_samples
            kl = kl*self.n_training_samples/math.sqrt(target_cats.shape[0]+1)
            # kl = kl*self.n_training_samples/target_segments.shape[0]
            return dict(nll=nll, kl=kl)
        elif "category" in input_dict.keys():
            cls_logits = cls_logits.view(-1, self.n_samples, cls_logits.size(1))
            mean_cls_logits = torch.mean(cls_logits, dim=1)
            nll = self.criteria(mean_cls_logits, input_dict["category"])
            kl, entropy = self.kl_and_entropy()
            kl = kl - self.entropy_weight * entropy
            return dict(nll=nll, kl=kl, cls_logits=mean_cls_logits)
        else:
            cls_logits = cls_logits.view(-1, self.n_samples, cls_logits.size(1))
            mean_cls_logits = torch.mean(cls_logits, dim=1)
            # predictive = point_wise_entropy(seg_logits, type='predictive')
            # aleatoric = point_wise_entropy(seg_logits, type='aleatoric')
            # epistemic = predictive - aleatoric
            return dict(cls_logits=mean_cls_logits)