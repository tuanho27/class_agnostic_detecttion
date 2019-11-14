from socket import gethostname
import os
conv_cfg = dict(type='ConvWS')
an_norm_cfg = dict(type='AN', K = 10, momentum=0.1, running = True, requires_grad=False)
norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)

cfg_name = os.path.basename(__file__).replace('.py','').split('/')[-1]
version =1
print(f'traning config {cfg_name}, version {version}')

# runtime settings
if 'X399' in gethostname():
	work_dir = f'/home/chuong/Workspace/Experiments/cloud/{cfg_name}/v{version}'
	data_root = '/home/chuong/Workspace/dataset/cloud/'
	load_from = None
	resume_from = None
	imgs_per_gpu = 2
else:
	work_dir = f'/home/member/Workspace/chuong/Experiments/cloud/{cfg_name}/v{version}'
	data_root =  '/home/member/Workspace/chuong/dataset/cloud/'
	load_from = None
	resume_from = None
	imgs_per_gpu = 12

print(f'WORKDIR: {work_dir}')
total_epochs = 20


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
	policy='Cosine',
	by_epoch=False,
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=1.0 / 3,
	target_lr=1e-4)
checkpoint_config = dict(interval=4)

# model settings
model = dict(
	type='HybridTaskCascade',
	num_stages=3,
	pretrained='open-mmlab://resnext50_32x4d',
	interleaved=True,
	mask_info_flow=True,
	backbone=dict(
		type='ResNeXt',
		depth=50,
		groups=32,
		base_width=4,
		num_stages=4,
		out_indices=(0, 1, 2, 3),
		frozen_stages=1,
		style='pytorch',
		gcb=dict(
			ratio=1./16.,
		),
		norm_cfg=norm_cfg,
		stage_with_gcb=(False, True, True, True),
		# atten_norm_cfg=an_norm_cfg,
		),
	neck=dict(
		type='NASFPN',
		in_channels=[256, 512, 1024, 2048],
		out_channels=256,
		start_level=1,
		add_extra_convs=True,
		num_outs=5),
	rpn_head=dict(
		type='RPNHead',
		in_channels=256,
		feat_channels=256,
		# anchor_scales=[8],
		# anchor_strides=[4, 8, 16, 32, 64],
		anchor_scales=[8],
		anchor_strides=[8, 16, 32, 64, 128],
		anchor_ratios=[0.5, 1.0, 2.0],
		target_means=[.0, .0, .0, .0],
		target_stds=[1.0, 1.0, 1.0, 1.0],
		loss_cls=dict(
			type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
	bbox_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
		out_channels=256,
		# featmap_strides=[4, 8, 16, 32]),
		featmap_strides=[8, 16, 32, 64]),
	bbox_head=[
		dict(
			type='SharedFCBBoxHead',
			num_fcs=2,
			in_channels=256,
			fc_out_channels=1024,
			roi_feat_size=7,
			num_classes=5,
			target_means=[0., 0., 0., 0.],
			target_stds=[0.1, 0.1, 0.2, 0.2],
			reg_class_agnostic=True,
			loss_cls=dict(
				type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
			loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
		dict(
			type='SharedFCBBoxHead',
			num_fcs=2,
			in_channels=256,
			fc_out_channels=1024,
			roi_feat_size=7,
			num_classes=5,
			target_means=[0., 0., 0., 0.],
			target_stds=[0.05, 0.05, 0.1, 0.1],
			reg_class_agnostic=True,
			loss_cls=dict(
				type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
			loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
		dict(
			type='SharedFCBBoxHead',
			num_fcs=2,
			in_channels=256,
			fc_out_channels=1024,
			roi_feat_size=7,
			num_classes=5,
			target_means=[0., 0., 0., 0.],
			target_stds=[0.033, 0.033, 0.067, 0.067],
			reg_class_agnostic=True,
			loss_cls=dict(
				type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
			loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
	],
	mask_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
		out_channels=256,
		# featmap_strides=[4, 8, 16, 32]),
		featmap_strides=[8, 16, 32, 64]),
	mask_head=dict(
		type='HTCMaskHead',
		num_convs=4,
		in_channels=256,
		conv_out_channels=256,
		num_classes=5,
		loss_mask=dict(
			type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
)
# model training and testing settings
train_cfg = dict(
	rpn=dict(
		assigner=dict(
			type='MaxIoUAssigner',
			pos_iou_thr=0.7,
			neg_iou_thr=0.3,
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
		nms_pre=1000,
		nms_post=1000,
		max_num=1000,
		nms_thr=0.7,
		min_bbox_size=0),
	rcnn=[
		dict(
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
			mask_size=28,
			pos_weight=-1,
			debug=False),
		dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.6,
				neg_iou_thr=0.6,
				min_pos_iou=0.6,
				ignore_iof_thr=-1),
			sampler=dict(
				type='RandomSampler',
				num=512,
				pos_fraction=0.25,
				neg_pos_ub=-1,
				add_gt_as_proposals=True),
			mask_size=28,
			pos_weight=-1,
			debug=False),
		dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.7,
				neg_iou_thr=0.7,
				min_pos_iou=0.7,
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
	],
	stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
	rpn=dict(
		nms_across_levels=False,
		nms_pre=1000,
		nms_post=1000,
		max_num=1000,
		nms_thr=0.7,
		min_bbox_size=0),
	rcnn=dict(
		score_thr=0.001,
		nms=dict(type='nms', iou_thr=0.5),
		max_per_img=100,
		mask_thr_binary=0.5),
	keep_all_stages=False)

# dataset settings
dataset_type = 'CloudDataset'

img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
	dict(
		type='ShiftScaleRotate',
		shift_limit=0.0625,
		scale_limit=0.0,
		rotate_limit=0,
		interpolation=1,
		p=0.5),
	dict(
		type='RandomBrightnessContrast',
		brightness_limit=[0.1, 0.3],
		contrast_limit=[0.1, 0.3],
		p=0.2),
	dict(
		type='OneOf',
		transforms=[
			dict(
				type='RGBShift',
				r_shift_limit=10,
				g_shift_limit=10,
				b_shift_limit=10,
				p=1.0),
			dict(
				type='HueSaturationValue',
				hue_shift_limit=20,
				sat_shift_limit=30,
				val_shift_limit=20,
				p=1.0)
		],
		p=0.1),
	# dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
	# dict(type='ChannelShuffle', p=0.1),
	dict(
		type='OneOf',
		transforms=[
			dict(type='Blur', blur_limit=3, p=1.0),
			dict(type='MedianBlur', blur_limit=3, p=1.0)
		],
		p=0.1),
	dict(type='VerticalFlip',p=0.5),
	dict(type='CoarseDropout',
		 max_holes=8, max_height=50, max_width=100,
		 min_holes=2, min_height=8, min_width=8, p=0.2)
]

train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
	dict(type='Resize', img_scale=(1050, 700), keep_ratio=True),
	dict(type='Pad', size_divisor=32),
	dict(type='Albu',
		transforms=albu_train_transforms,
		bbox_params=dict(
			type='BboxParams',
			format='pascal_voc',
			label_fields=['gt_labels'],
			min_visibility=0.0,
			filter_lost_elements=True),
		keymap={
			'img': 'image',
			'gt_masks': 'masks',
			'gt_bboxes': 'bboxes'
		},
		update_pad_shape=False,
		skip_img_without_anno=True),
	# dict(type='ObjDetAugmentation',policy='v0'),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(1050, 700),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(type='RandomFlip', flip_ratio=0.5),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']),
		])
]
data = dict(
	imgs_per_gpu=imgs_per_gpu,
	workers_per_gpu=8,
	train=dict(type='RepeatDataset', times=16,
		dataset=dict(
			type=dataset_type,
			ann_file=data_root + 'train_ann.pickle',
			img_prefix=data_root + 'train_images/',
			pipeline=train_pipeline)),
	val=dict(
		type=dataset_type,
		ann_file=data_root + 'train_ann.pickle',
		img_prefix=data_root + 'train_images/',
		pipeline=test_pipeline),
	test=dict(
		type=dataset_type,
		ann_file=data_root + 'train_ann.pickle',
		img_prefix=data_root + 'train_images/',
		pipeline=test_pipeline))

# yapf:disable
log_config = dict(
	interval=50,
	hooks=[
		dict(type='TextLoggerHook'),
		dict(type='TensorboardLoggerHook')
	])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
