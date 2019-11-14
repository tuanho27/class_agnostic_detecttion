# %%
from mmdet.datasets import CloudDataset
# from mmdet.datasets.builder import build_dataset
import mmcv
import matplotlib.pyplot as plt
import numpy as np
from mmdet.datasets.visualize_ground_truth import show_gt, visualize_gt
from mmdet.apis import show_result
import random

data_root = '/home/chuong/Workspace/dataset/cloud/'

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
ground_truth = (gt_labels.numpy(),gt_bboxes.numpy(),gt_masks)
filename = img_info['filename'].replace('.jpg','')
classes_name = ['background']+list(data_org.CLASSES)
# show_gt(img_input, ground_truth, classes_name, color='red', font_scale=0.5, show=False, out_file=f'{filename}_org.png')
img_gt = show_gt(img_input, ground_truth, classes_name, color='red', font_scale=0.75, show=False)
# import pdb; pdb.set_trace()
# %%
# or save the visualization results to image files
result_file = '/home/chuong/Workspace/Experiments/cloud/retina_x50gcb_nasfpn_cloud/v1/epoch_24_025.pkl'
det_results = mmcv.load(result_file)
pred_result = det_results[idx]
show_result(img_gt, pred_result, data_org.CLASSES, score_thr=0.25, show=False,out_file=f'{idx}_{filename}_pred.png')
