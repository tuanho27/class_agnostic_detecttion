# model settings
imgs_per_gpu=16
model = dict(
    type='FasterRCNNPair',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=21,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    siamese_matching_head=dict(
        type='SiameseMatching',
        in_channels=256,
        feat_channels=128,
        dist_mode='cosine',
        loss_siamese=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        ),
    relation_matching_head=dict(
        type='RelationMatching',
        in_channels=256,
        feat_channels=128,
        ))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6, #0.7 -> 0.6
            neg_iou_thr=0.4, # 0.3 -> 0.4
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=256,
        nms_post=128,
        max_num=128,
        nms_thr=0.7, #0.7 -> 0.5
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=256,
        nms_post=64,
        max_num=64,
        nms_thr=0.6,
        min_bbox_size=0),
    topk_pair_select=100,
    mode='infer',  ## infer | eval
    match_head='relation', ## siamese | relation
    score_thr=0.95,
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'VOCPairDataset'
# dataset_type='VOCDataset'
data_root = '/home/member/Workspace/dataset/VOC/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(900, 600), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(900, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=train_pipeline,
        txt_file = './ccdetection/configs/faster_rcnn/list_pairs_img_voc2007.txt',
        txt_eval_file = './ccdetection/configs/faster_rcnn/list_pairs_img_test_voc2007.txt'), 
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007TEST/val.txt',
        img_prefix=data_root + 'VOC2007TEST/',
        pipeline=test_pipeline,
        txt_file = './ccdetection/configs/faster_rcnn/list_pairs_img_voc2007.txt',
        txt_eval_file = './ccdetection/configs/faster_rcnn/list_pairs_img_test_voc2007.txt'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007TEST/val.txt',
        img_prefix=data_root + 'VOC2007TEST/',
        pipeline=test_pipeline,
        txt_file = './ccdetection/configs/faster_rcnn/list_pairs_img_voc2007.txt',
        txt_eval_file = './ccdetection/configs/faster_rcnn/list_pairs_img_test_voc2007.txt'))

evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[3])  # actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[5, 7, 11])
    
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 15  # actual epoch = 4 * 3 = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_voc0712_update_data'
# load_from = None
load_from = './work_dirs/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
resume_from = None
workflow = [('train', 1)]
