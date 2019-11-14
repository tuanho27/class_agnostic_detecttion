dataset_type='MPIIDataset'
data_root='/home/cybercore/Workspace/dataset/mpii/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPoseAnnotations', with_joints=True, with_heatmap=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='Pad', size_divisor=32),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='DefaultFormatBundle'),
    # dict(
    #     type='Collect',
    #     keys=['img', 'gt_joints', 'gt_heatmap', 'gt_masks'],
    #     meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
    #                'pad_shape', 'scale_factor'))
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annot/train_multi.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annot/valid_multi.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annot/val_multi.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline))