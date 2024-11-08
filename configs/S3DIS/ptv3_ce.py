_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = None
dynamic_batching = True
max_points_per_batch=307200
num_worker = 8
num_worker_test = 2
mix_prob = 0.8
empty_cache = True
enable_amp = True
epoch = 200 # train (epoch/eval_epoch) epochs and then eval for one epoch
eval_epoch = 20
seed = 25326354

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=13,
    backbone_out_channels=32,
    stochastic=False,
    backbone=dict(
        type="PT-v3",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(16, 16, 16, 16, 16),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(32, 64, 128, 256),
        dec_num_head=(2, 4, 8, 16),
        dec_patch_size=(16, 16, 16, 16),
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
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
        dict(type="CrossEntropyLoss", loss_weight=1., ignore_index=-1),
        dict(type="TverskyLoss", loss_weight=1., ignore_index=-1),
    ],
)

# scheduler settings
optimizer = dict(type="Adam", lr=0.01, weight_decay=5e-4)
milestone_ratios = [0.6, 0.8]

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
        type=dataset_type,
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
            dict(type="SphereCrop", point_max=102400, mode="random"),
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
    dict(type="DynamicBatchSizeProfiler"),
    # dict(type="GPUMemoryInspector"),
    dict(type="DynamicInformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

train = dict(type="DynamicTrainer")

test = dict(type='SemSegTester', verbose=True)