
# fp16 settings
fp16 = dict(loss_scale=512.)


debug = True
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 30])
data_root = 'dataset-coco/'
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fcos_mask_r50_fp16'
load_from = None
# resume_from = 'model_zoo/fcos_r50_caffe_fpn_1x_4gpu_20190516-a7cac5ff.pth'
# resume_from = 'model_zoo/fcos_mstrain_640_800_r50_caffe_fpn_gn_2x_4gpu_20190516-f7329d80.pth'
resume_from = None#f'{work_dir}/latest.pth'
workflow = [('train', 1)]
num_samples = None

train_ann_file = data_root + 'annotations/instances_train2017.json'
train_img_dir = data_root+'images/train2017/'
roi_out_size = 14
imgs_per_gpu = 12
if debug:
    imgs_per_gpu=1
    # work_dir += '/debug'
    total_epochs = 501
    checkpoint_config = dict(interval=100)
    num_samples = 1
    workers_per_gpu = 1
    imgs_per_gpu = 1
    train_ann_file = data_root + 'annotations/instances_val2017.json'
    train_img_dir = data_root+'images/val2017/'
    # learning policy

    log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook'),
            # dict(type='TensorboardLoggerHook')
        ])
    lr_config = dict(
        policy='step',
        warmup='constant',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[total_epochs-50])

# model settings
model = dict(
    type='FCOSMask',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),

    # FOR MASK HEAD
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=roi_out_size, sample_num=2),
        out_channels=256,
        featmap_strides=[8,16, 32],
        finest_scale=112),

    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=81,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
    )

)

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    rpn_proposal=dict(
        nms_pre=1000,
        min_bbox_size=10,
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100
    ),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            min_pos_iou=0.5,
            neg_iou_thr=0.4,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False)
)



test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100,
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))



#--------------------------- dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_dir,
        pipeline=train_pipeline, num_samples=num_samples),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline, num_samples=num_samples),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline, num_samples=num_samples))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.015,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))

# optimizer = dict(
#     type='Adam',
#     lr=0.0001
# )
optimizer_config = dict(grad_clip=None)
# learning policy
# 