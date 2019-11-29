from mmdet.models.backbones import timm_channel_pyramid
debug=True
work_dir = './work_dirs/polar_b1'
data_root = './datasets/coco/'


##
# optimizer
lr_ratio = 1

optimizer = dict(
    type='SGD',
    lr=0.01 * lr_ratio,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3 / lr_ratio,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
train_ann_file = data_root + 'annotations/instances_train2017.json'
train_img_dir = data_root+'images/train2017/'
num_samples=None
imgs_per_gpu=4
repeat=1
if debug:
    num_samples=1
    repeat=50
    imgs_per_gpu=1
    # optimizer
    train_ann_file = data_root + 'annotations/instances_val2017.json'
    train_img_dir = data_root+'images/val2017/'
    optimizer = dict(
        type='Adam',
        lr=0.001 * lr_ratio)
    # learning policy
    lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3 / lr_ratio,
        step=[8, 11])
    checkpoint_config = dict(interval=10)
    # yapf:disable
    log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook'),
            # dict(type='TensorboardLoggerHook')
        ])
    # yapf:enable
    # runtime settings
    total_epochs = 12
    device_ids = range(4)
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    load_from = None
    resume_from = None




##





# model settings
wh_ratio=1333/800
EfficientDetConfig ={
	'D0': dict(Backbone='efficientnet_b0',ImgSize=(896, 512),  fpn_channel=64, fpn_stack=2,head_depth=3),
	'D1': dict(Backbone='efficientnet_b1',ImgSize=(1024, 640),  fpn_channel=88, fpn_stack=3,head_depth=3),
	'D2': dict(Backbone='efficientnet_b2',ImgSize=(1280, 768),  fpn_channel=112,fpn_stack=4,head_depth=3),
	'D3': dict(Backbone='efficientnet_b3',ImgSize=(1408, 896),  fpn_channel=160,fpn_stack=5,head_depth=4),
	'D4': dict(Backbone='efficientnet_b4',ImgSize=(1024*wh_ratio, 1024),fpn_channel=224,fpn_stack=6,head_depth=4),
	'D5': dict(Backbone='efficientnet_b5',ImgSize=(1280*wh_ratio, 1280),fpn_channel=288,fpn_stack=7,head_depth=4),
	'D6': dict(Backbone='efficientnet_b6',ImgSize=(1408*wh_ratio, 1408),fpn_channel=384,fpn_stack=8,head_depth=5),
}
model_cfg=EfficientDetConfig['D1']

model = dict(
    type='PolarMask',
    pretrained=None,
    backbone=dict(
        type='TimmCollection',
        model_name=model_cfg['Backbone'],
        drop_rate=0.0,
        norm_eval=True,
        pretrained=True),
	neck=dict(
		type='StackBiFPN',
		in_channels=timm_channel_pyramid[model_cfg['Backbone']],
		out_channels=model_cfg['fpn_channel'],
		start_level=1,
		num_outs=5,
		fpn_stack=model_cfg['fpn_stack'],
		fpn_conv_groups=model_cfg['fpn_channel'], #Use DepthWise
		add_extra_convs=True,
	),
    bbox_head=dict(
        type='PolarMask_Head',
        num_classes=81,
        in_channels=model_cfg['fpn_channel'],
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
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
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
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'Coco_Seg_Dataset'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_dir,
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        # size_divisor=0,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False
        , num_samples=num_samples,repeat=repeat),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        # size_divisor=0,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False
        , num_samples=num_samples),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1280, 768),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        resize_keep_ratio=False,
        test_mode=True, 
        num_samples=num_samples)
)
