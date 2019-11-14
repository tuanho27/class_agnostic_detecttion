# %%
from mmdet.datasets import CloudDataset
# from mmdet.datasets.builder import build_dataset
import mmcv
import matplotlib.pyplot as plt
import numpy as np
from mmdet.datasets.visualize_ground_truth import show_gt, visualize_gt
import random

data_root = '/home/chuong/Workspace/dataset/cloud/'

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
	dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
	# dict(type='ChannelShuffle', p=0.1),
	# dict(
	# 	type='OneOf',
	# 	transforms=[
	# 		dict(type='Blur', blur_limit=3, p=1.0),
	# 		dict(type='MedianBlur', blur_limit=3, p=1.0)
	# 	],
	# 	p=0.1),
	dict(type='VerticalFlip',p=0.5),
	dict(type='CutOut',num_holes=8, max_h_size=50, max_w_size=100,p=0.2)

]

train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
	dict(type='Resize', img_scale=(1050, 700), keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0),
	dict(type='Normalize', **img_norm_cfg),
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
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

orig_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
	dict(type='Resize', img_scale=(1050, 700), keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0),
	dict(type='Normalize',  mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]


data_train=CloudDataset(ann_file=data_root + 'train_ann.pickle',
						img_prefix=data_root + 'train_images',
						pipeline=train_pipeline)

data_org=CloudDataset(ann_file=data_root + 'train_ann.pickle',
						img_prefix=data_root + 'train_images',
						pipeline=orig_pipeline)

# %%
# Inspect One sample
idx = random.randint(0,100)
# idx=0
img_info = data_org.img_infos[idx]
ann_info = data_org.get_ann_info(idx)
sample = data_org[idx]
img_meta = sample['img_meta'].data
img = sample['img'].data
gt_bboxes= sample['gt_bboxes'].data
gt_labels= sample['gt_labels'].data
gt_masks = sample['gt_masks'].data

if False:
	print(img_info)
	print(ann_info)
	print(sample.keys())
	print('img_meta')
	print(img_meta)
	print('gt_labels.shape=',gt_labels.shape)
	print(gt_labels)
	print('gt_bboxes.shape=',gt_bboxes.shape)
	print(gt_bboxes)
	print('gt_masks.shape=',gt_masks.shape)

#retrieve input image
img_input = img.permute(1,2,0).numpy()
result = (gt_labels.numpy(),gt_bboxes.numpy(),gt_masks)
filename = img_info['filename'].replace('.jpg','')
classes_name = ['background']+list(data_train.CLASSES)
img_out = show_gt(
			img_input, result, classes_name, show=False, out_file=f'{filename}_org.png')

# %%
aug_sample = data_train[idx]
aug_img_meta = aug_sample['img_meta'].data
aug_img = aug_sample['img'].data
aug_gt_bboxes= aug_sample['gt_bboxes'].data
aug_gt_labels= aug_sample['gt_labels'].data
aug_gt_masks = aug_sample['gt_masks'].data
# # visualize_gt(img_input,gt_labels.numpy(),gt_bboxes.numpy(),gt_masks,classes_name)
mean = aug_img_meta['img_norm_cfg']['mean']
std = aug_img_meta['img_norm_cfg']['std']
aug_img_input = mmcv.rgb2bgr(aug_img.permute(1,2,0).numpy()*std+mean)
aug_result = (aug_gt_labels.numpy(),aug_gt_bboxes.numpy(),aug_gt_masks)
aug_img_out = show_gt(
			aug_img_input, aug_result, classes_name, show=False, out_file=f'{filename}_aug.png')