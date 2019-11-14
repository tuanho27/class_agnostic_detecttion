# %%
from mmdet.datasets.coco import CocoDataset
import mmcv
import matplotlib.pyplot as plt
import numpy as np
from mmdet.datasets.visualize_ground_truth import show_gt, visualize_gt

data_root = '/home/chuong/Workspace/dataset/coco/'

img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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

data_train=CocoDataset(ann_file=data_root + 'annotations/instances_train2017.json',
						img_prefix=data_root + 'train2017/',
						pipeline=train_pipeline)

# %%
# Inspect One sample
idx = 0 
img_info = data_train.img_infos[idx]
ann_info = data_train.get_ann_info(idx)
sample = data_train[idx]
img_meta = sample['img_meta'].data
img = sample['img'].data
gt_bboxes= sample['gt_bboxes'].data
gt_labels= sample['gt_labels'].data
gt_masks = sample['gt_masks'].data

# %%
mean = img_meta['img_norm_cfg']['mean']
std = img_meta['img_norm_cfg']['std']
#retrieve inpyt image
img_input = mmcv.rgb2bgr(img.permute(1,2,0).numpy()*std+mean)
result = (gt_labels.numpy(),gt_bboxes.numpy(),gt_masks)
filename = img_info['id']
classes_name = ['background']+list(data_train.CLASSES)
img_out = show_gt(
			img_input, result, classes_name, show=False, out_file=f'test_{filename}.png')
visualize_gt(img_input,gt_labels.numpy(),gt_bboxes.numpy(),gt_masks,classes_name)

