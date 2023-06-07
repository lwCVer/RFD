_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

model = dict(
    type='S2ANet',
    pretrained='/data1/lw/RFD_V2/OBBDetection/backbone_checkpoints/cswin_tiny_init.pth',
    backbone=dict(
        type='CSWin',
        embed_dim=64,
        depth=[1, 1, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        drop_path_rate=0.3,
        use_chk=False
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='S2AHead',
        feat_channels=256,
        align_type='AlignConv',
        heads=[
            dict(
                type='ODMHead',
                num_classes=15,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                anchor_generator=dict(
                    type='Theta0AnchorGenerator',
                    scales=[4],
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            ),
            dict(
                type='ODMHead',
                num_classes=15,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                with_orconv=True,
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            )
        ]
    )
)

# training and testing settings
train_cfg = [
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
]
test_cfg = dict(
    skip_cls=[True, False],
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=2000)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)
