from socket import gethostname

# Server adaptation
if 'X399' in gethostname():
	imgs_per_gpu = 2
	total_epochs = 12
	load_from = None
	resume_from = None
	pretrained = 'torchvision://resnet50'
	data_root = '/home/cybercore/Workspace/dataset/coco/'
	work_dir = '/home/cybercore/thuync/checkpoints/retinamask_r50_newloss/'
	fp16 = dict(loss_scale=512.)

elif '184' in gethostname():
	imgs_per_gpu = 16
	total_epochs = 12
	resume_from = None
	pretrained = 'torchvision://resnet50'
	data_root= '/home/member/Workspace/dataset/coco/'
	work_dir = '/home/member/Workspace/thuync/checkpoints/retinamask_r50_newloss/'
	load_from = "/home/member/Workspace/thuync/checkpoints/retinanet_r50/retinanet_r50_fpn_1x_20181125-7b0c2548.pth"
	fp16 = dict(loss_scale=512.)

elif '185' in gethostname():
	lr_start = 1e-2
	lr_end = 1e-4
	imgs_per_gpu = 16
	total_epochs = 12
	resume_from = None
	pretrained = 'torchvision://resnet50'
	data_root= '/home/member/Workspace/dataset/coco/'
	work_dir = '/home/member/Workspace/thuync/checkpoints/retinamask_r50_newloss/'
	load_from = "/home/member/Workspace/thuync/checkpoints/retinanet_r50/retinanet_r50_fpn_1x_20181125-7b0c2548.pth"
	fp16 = dict(loss_scale=512.)

elif '186' in gethostname():
	imgs_per_gpu = 16
	total_epochs = 12
	load_from = None
	resume_from = None
	pretrained = 'torchvision://resnet50'
	data_root= '/home/user/thuync/datasets/coco/'
	work_dir = '/home/user/thuync/checkpoints/retinamask_r50_newloss/'
	fp16 = dict(loss_scale=512.)

# model settings
model = dict(
	type='RetinaMask',
	pretrained=pretrained,
	backbone=dict(
		type='ResNet',
		depth=50,
		num_stages=4,
		out_indices=(0, 1, 2, 3),
		frozen_stages=1,
		style='pytorch',
	),
	neck=dict(
		type='FPN',
		in_channels=[256, 512, 1024, 2048],
		out_channels=256,
		start_level=1,
		add_extra_convs=True,
		num_outs=5,
	),
	bbox_head=dict(
		type='RetinaHead',
		num_classes=81,
		in_channels=256,
		stacked_convs=4,
		feat_channels=256,
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
		out_channels=256,
		featmap_strides=[4, 8, 16, 32],
	),
	mask_head=dict(
		type='FCNMaskHead',
		num_convs=4,
		in_channels=256,
		conv_out_channels=256,
		num_classes=81,
		# loss_mask=dict(type='SegmFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
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
	dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
	workers_per_gpu=8,
	train=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_train2017.json',
		img_prefix=data_root + 'images/train2017/',
		pipeline=train_pipeline,
	),
	val=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_val2017.json',
		img_prefix=data_root + 'images/val2017/',
		pipeline=test_pipeline,
	),
	test=dict(
		type=dataset_type,
		ann_file=data_root + 'annotations/instances_val2017.json',
		img_prefix=data_root + 'images/val2017/',
		pipeline=test_pipeline,
	),
)
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
