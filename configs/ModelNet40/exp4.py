_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 30 # total batch_size in all gpus
num_worker = 8  # total num_workers in all gpus
num_worker_test = 4
batch_size_val = 30
batch_size_test = 30
empty_cache = True 
enable_amp = True # enable automatic mixed precision
epoch = 20  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 5  # sche total eval & checkpoint epoch

# model settings
model = dict(
    type="BayesClassifier",
    num_classes=40,
    backbone_embed_dim=512,
    n_components=4,
    n_training_samples=1,
    n_samples=4,
    stochastic=True,
    stochastic_modules=[],
    prior_mean=1.0, 
    prior_std=0.1, 
    post_mean_init=(1.0, 0.1), 
    post_std_init=(0.1, 0.05),
    kl_weight_init=1e-2,
    kl_weight_final=1e-1,
    entropy_weight=1.,
    backbone=dict(
        type="PT-BNN",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(16, 16, 16, 16, 16),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(16, 16, 16, 16),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=True,
        
        stochastic_modules=['proj'],
        n_components=4,
        prior_mean=1.0,
        prior_std=0.1, 
        post_mean_init=(1.0, 0.1), 
        post_std_init=(0.1, 0.05),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# train settings
# optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
# scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
optimizer = dict(type="Adam", lr=0.001, weight_decay=0.00)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "ModelNetDataset"
data_root = "/userhome/cs2/yanniki/capstone/BayesianPointTransformer/data/modelnet40_normal_resampled"
cache_data = False
class_names = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

data = dict(
    num_classes=40,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=10000, mode="random"),
            # dict(type="CenterShift", apply_z=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                keys=("coord", "normal"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "normal"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
        ],
        test_mode=True,
        test_cfg=dict(
            voting=False,
            post_transform=[
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "normal"),
                    return_grid_coord=True,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "category"),
                    feat_keys=["coord", "normal"],
                ),
            ],
            aug_transform=[
                # [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],  # 1
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 2
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 3
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 4
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 5
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 5
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 6
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 7
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 8
                # [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],  # 9
            ],
        ),
    ),
)

# hooks
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="BayesClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

train = dict(type="DefaultTrainer")
# tester
# test = dict(type="ClsVotingTester", num_repeat=2)
test = dict(type="BayesClsTester", verbose=False)