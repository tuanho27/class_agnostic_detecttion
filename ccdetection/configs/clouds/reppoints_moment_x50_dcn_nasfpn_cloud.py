from socket import gethostname
import os
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

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
total_epochs = 24
auto_resume = True

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
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2)

#model setting
model = dict(
    type='RepPointsDetector',
    pretrained='open-mmlab://resnext50_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        gcb=dict(
			ratio=1./16.,
		),
        stage_with_gcb=(False, True, True, True),
        dcn=dict(
            modulated=False,
            groups=32,
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),),
    neck=dict(
		type='NASFPN',
		in_channels=[256, 512, 1024, 2048],
		out_channels=256,
		start_level=1,
		add_extra_convs=True,
		num_outs=5,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=5,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        loss_cls=dict(type='AutoFocalLoss',use_sigmoid=True,gamma=2.0,alpha=0.25,loss_weight=1.0),
        loss_bbox_init=dict(type='AdaptiveRobustLoss_R', num_dims=4, loss_weight=0.5),
        loss_bbox_refine=dict(type='AdaptiveRobustLoss_R', num_dims=4, loss_weight=1.0),
        # loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        # loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=100,
    score_thr=0.25,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=50)

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
		type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
	dict(type='Resize', img_scale=(1050, 700), keep_ratio=True),
	dict(type='Pad', size_divisor=32),
	# dict(type='Albu',
	# 	transforms=albu_train_transforms,
	# 	bbox_params=dict(
	# 		type='BboxParams',
	# 		format='pascal_voc',
	# 		label_fields=['gt_labels'],
	# 		min_visibility=0.0,
	# 		filter_lost_elements=True),
	# 	keymap={
	# 		'img': 'image',
	# 		'gt_masks': 'masks',
	# 		'gt_bboxes': 'bboxes'
	# 	},
	# 	update_pad_shape=False,
	# 	skip_img_without_anno=True),
	dict(type='ObjDetAugmentation',policy='v0'),
    dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
			preload_mask=False,
			ann_file=data_root + 'train_ann.pickle',
			img_prefix=data_root + 'train_images/',
			pipeline=train_pipeline)),
	val=dict(
		type=dataset_type,
		preload_mask=False,
		with_ignore_bboxes=False,
		ann_file=data_root + 'train_ann.pickle',
		img_prefix=data_root + 'train_images/',
		pipeline=test_pipeline),
	test=dict(
		type=dataset_type,
		preload_mask=False,
		with_ignore_bboxes=False,
		ann_file=data_root + 'train_ann.pickle',
		img_prefix=data_root + 'train_images/',
		pipeline=test_pipeline))

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        # dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
