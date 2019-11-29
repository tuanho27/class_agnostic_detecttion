from mmdet.models.backbones import timm_channel_pyramid
debug=True



use_gn = False
lr_start = 1e-2
lr_end = 1e-4
imgs_per_gpu = 8
total_epochs = 12
resume_from = None
pretrained = None
img_scale = (1280, 768)
data_root = 'datasets/coco/'
work_dir = 'work_dirs/polar_b1_bifpn'
load_from = None#'/home/member/Workspace/phase3/polar-B1-FPN/epoch_9.pth'
fp16 = dict(loss_scale=512.)







step = [8, 11]
log_interval = 20
warmup_iters = 500
num_samples = None
size_divisor = 128
workers_per_gpu = 6
ckpt_interval = 1
train_ann_file = data_root + 'annotations/instances_train2017.json'
val_ann_file = data_root + 'annotations/instances_val2017.json'
test_ann_file = data_root + 'annotations/instances_val2017.json'
train_img_prefix = data_root + 'images/train2017/'
val_img_prefix = data_root + 'images/val2017/'
test_img_prefix = data_root + 'images/val2017/'

if debug:
	log_interval = 1
	warmup_iters = 1
	total_epochs = 30
	ckpt_interval = total_epochs
	num_samples = 1
	workers_per_gpu = 1
	step = [int(0.5*total_epochs), int(0.75*total_epochs)]
	train_ann_file = data_root + 'annotations/instances_val2017.json'
	train_img_prefix = data_root + 'images/val2017/'

# model settings
if use_gn:
	conv_cfg = dict(type='ConvWS')
	norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
else:
	conv_cfg = None
	norm_cfg = None
    
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
	),
	# neck=dict(
	# 	type='FPN',
	# 	in_channels=[24, 40, 112, 320],
	# 	out_channels=256,
	# 	start_level=1,
	# 	add_extra_convs=True,
	# 	extra_convs_on_inputs=False,  # use P5
	# 	num_outs=5,
	# 	relu_before_extra_convs=True,
	# ),

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
			type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
	),
	semseg_head=dict(
		type='SemSegHead',
		num_convs=4,
		in_channels=model_cfg['fpn_channel'],
		conv_kernel_size=3,
		conv_out_channels=256,
		input_index=0,
		upsample_method='bilinear',
		upsample_ratio=2,
		num_classes=1,
		loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
	),
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
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
	dict(type='Resize', img_scale=img_scale, keep_ratio=False),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=size_divisor),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=img_scale,
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=False),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=size_divisor),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']),
		])
]
data = dict(
	imgs_per_gpu=imgs_per_gpu,
	workers_per_gpu=workers_per_gpu,
	train=dict(
		type=dataset_type,
		ann_file=train_ann_file,
		img_prefix=train_img_prefix,
		img_scale=img_scale,
		num_samples=num_samples,
		img_norm_cfg=img_norm_cfg,
		size_divisor=size_divisor,
		flip_ratio=0.5,
		with_mask=True,
		with_fg_mask=True,
		seg_scale_factor=0.25,
		with_crowd=False,
		with_label=True,
		resize_keep_ratio=False),
	val=dict(
		type=dataset_type,
		ann_file=val_ann_file,
		img_prefix=val_img_prefix,
		img_scale=img_scale,
		img_norm_cfg=img_norm_cfg,
		size_divisor=size_divisor,
		flip_ratio=0,
		with_mask=False,
		with_crowd=False,
		with_label=True,
		resize_keep_ratio=False),
	test=dict(
		type=dataset_type,
		ann_file=test_ann_file,
		img_prefix=test_img_prefix,
		img_scale=img_scale,
		img_norm_cfg=img_norm_cfg,
		size_divisor=size_divisor,
		flip_ratio=0,
		with_mask=False,
		with_crowd=False,
		with_label=False,
		resize_keep_ratio=False,
		test_mode=True))

# optimizer
optimizer = dict(
	type='SGD',
	lr=lr_start,
	momentum=0.9,
	weight_decay=0.0001,
	paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
	# policy='step', step=step,
	policy='cosine', target_lr=lr_end, by_epoch=False,
	warmup='linear', warmup_iters=warmup_iters, warmup_ratio=1.0 / 3 / (lr_start/1e-2),
)
checkpoint_config = dict(interval=ckpt_interval)

# yapf:disable
log_config = dict(
	interval=log_interval,
	hooks=[
		dict(type='TextLoggerHook'),
		# dict(type='TensorboardLoggerHook')
	])

# yapf:enable
# runtime settings
device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
