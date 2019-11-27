from mmdet.models.backbones import timm_channel_pyramid
# """
# 'efficientnet_b0':[24,40,112,320],
# 'efficientnet_b1':[24,40,112,320],
# 'efficientnet_b2':[24,48,120,352],
# 'efficientnet_b3':[32,48,136,384],
# 'efficientnet_b4':[32,56,160,448],
# 'efficientnet_b5':[40,64,176,512],
# 'efficientnet_b6':[40,72,200,576],
# 'efficientnet_b7':[48,80,224,640],
# 'mixnet_s':[24,40,120,200],
# 'mixnet_m':[32,40,120,200],
# 'mixnet_l':[40,56,160,264],
# """



# fp16 = dict(loss_scale=512.)
# debug
debug=True
num_samples = None
workers_per_gpu = 2
if debug:
    num_samples = 200
    workers_per_gpu = 1
# fp16 settings

lr_start = 1e-2
lr_end = 1e-4
imgs_per_gpu = 2
total_epochs = 12
data_root= '/set/by/data_root/in/command_train/command_test'
work_dir = '/set/by/work_dir/in/command_train/command_test'
load_from = None #or '/set/by/load_from/in/command_train/command_test'
resume_from = None #or '/set/by/resume_from/in/command_train/command_test'
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
model_cfg=EfficientDetConfig['D0']
# model settings
model = dict(
	type='RetinaMask',
    backbone=dict(
        type='TimmCollection',
        model_name=model_cfg['Backbone'],
        drop_rate=0.1,
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
		type='RetinaHead',
		num_classes=81,
		in_channels=model_cfg['fpn_channel'],
		stacked_convs=model_cfg['head_depth'],
		feat_channels=model_cfg['fpn_channel'],
		octave_base_scale=4,
		scales_per_octave=3,
		anchor_ratios=[0.5, 1.0, 2.0],
		anchor_strides=[8, 16, 32, 64, 128],
		loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
		# loss_cls=dict(type='AutoFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.5, loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
		# loss_bbox=dict(type='AdaptiveRobustLoss_R', num_dims=4, loss_weight=1.0),
	),
	mask_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
		out_channels=model_cfg['fpn_channel'],
		featmap_strides=[8, 16, 32],
		finest_scale=112,
	),
	mask_head=dict(
		type='FCNMaskHead',
		num_convs=model_cfg['head_depth'],
		in_channels=model_cfg['fpn_channel'],
		conv_out_channels=model_cfg['fpn_channel'],
		num_classes=81,
		# loss_mask=dict(type='SegmFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
		loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
	),
)
# training and testing settings
train_cfg = dict(
	assigner=dict(
		type='MaxIoUAssigner',
		pos_iou_thr=0.5,
		neg_iou_thr=0.4,
		min_pos_iou=0,
		ignore_iof_thr=-1,
	),
	allowed_border=-1,
	pos_weight=-1,
	rpn_proposal=dict(
		nms_pre=1000,
		min_bbox_size=0,
		score_thr=0.0,
		nms=dict(type='nms', iou_thr=0.7),
		max_per_img=1000
	),
	rcnn=dict(
		assigner=dict(
			type='MaxIoUAssigner',
			pos_iou_thr=0.5,
			neg_iou_thr=0.5,
			min_pos_iou=0.5,
			ignore_iof_thr=-1,
		),
		sampler=dict(
			type='RandomSampler',
			num=512,
			pos_fraction=0.25,
			neg_pos_ub=-1,
			add_gt_as_proposals=True,
		),
		mask_size=28,
		pos_weight=-1,
		debug=False,
	),
)
test_cfg = dict(
	nms_pre=1000,
	min_bbox_size=0,
	score_thr=0.05,
	nms=dict(type='nms', iou_thr=0.5),
	max_per_img=100,
	mask_thr_binary=0.5,
)
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
	dict(type='Resize', img_scale=model_cfg['ImgSize'], keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=128),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=model_cfg['ImgSize'],
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
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017/',
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
optimizer = dict(type='SGD', lr=lr_start, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
	# policy='step', step=[8, 11],
	policy='cosine', target_lr=lr_end, by_epoch=False,
	warmup='linear', warmup_iters=500, warmup_ratio=1.0/3,
)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
	interval=20,
	hooks=[
		dict(type='TextLoggerHook'),
		# dict(type='TensorboardLoggerHook')
	])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
