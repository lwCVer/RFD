_base_ = [
    '../_base_/models/upernet_convnext_RFD_V2.py', '../_base_/datasets/isaid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
# model = dict(
#     # pretrained='/data/Segmentation/Swin-Transformer-Semantic-Segmentation/swin_tiny_patch4_window7_224.pth',
#     backbone=dict(
#         embed_dim=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         ape=False,
#     ),
model = dict(
    backbone=dict(
        type='ConvNeXt_RFD',
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=16,
        ignore_index=255
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=16,
        ignore_index=255
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)