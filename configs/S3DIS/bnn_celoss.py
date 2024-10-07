_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 32  # bs: total bs in all gpus
dynamic_batching = True
max_points_per_batch=100000
num_worker = 8  
mix_prob = 0.8
empty_cache = True
enable_amp = True
epoch = 300 # train (epoch/eval_epoch) epochs and then eval for one epoch
eval_epoch = 100

# model settings
model = dict(
    type="BayesSegmentor",
    num_classes=13,
    backbone_out_channels=16,
    n_components=4,
    n_samples=4,
    stochastic=True,
    prior_mean=1.0, 
    prior_std=0.10, 
    post_mean_init=(1.0, 0.10), 
    post_std_init=(0.10, 0.05),
    kl_weight_init=1e-2,
    kl_weight_final=1.0,
    entropy_weight=0.5,
    backbone=dict(
        type="PT-BNN",
        in_channels=6,
        order=("z", "z-trans", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 4, 2),
        enc_channels=(16, 32, 64, 128, 256),
        enc_num_head=(2, 2, 4, 4, 8),
        enc_patch_size=(64, 64, 64, 64, 64),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(16, 32, 64, 128),
        dec_num_head=(2, 2, 4, 4),
        dec_patch_size=(64, 64, 64, 64),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=0.75, ignore_index=-1),
        # dict(type="FocalLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="TverskyLoss", loss_weight=0.25, ignore_index=-1),
    ],
)

# scheduler settings
optimizer = dict(type="AdamW", lr=0.01, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.01, 0.001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# scheduler = dict(
#     type="CosineAnnealingLR",
# )

param_dicts = [dict(keyword="block", lr=0.001),
               ]

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis_processed"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=f'Dynamic{dataset_type}',
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=51200, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    # precise eval is disabled due to computation limitation
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(
            #     type="Copy",
            #     keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            # ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    # "origin_coord",
                    "segment",
                    # "origin_segment",
                ),
                offset_keys_dict=dict(offset="coord",),
                # offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "normal"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[],
        ),),
    )

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="DynamicIterationTimer", warmup_iter=2),
    # dict(type="GPUMemoryInspector"),
    dict(type="DynamicInformationWriter"),
    dict(type="BayesSemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

train = dict(type="DynamicTrainer")

test = dict(type='BayesSemSegTester', verbose=True)