# %%
from mmdet.datasets.builder import build_dataset
# from mmdet.models import builder
from mmdet.datasets.visualize_ground_truth import show_gt, visualize_gt
from mmcv import Config

import mmcv
import matplotlib.pyplot as plt
import numpy as np
import random

cfg_file= 'ccdetection/configs/clouds/htc_x50gcb_nasfpn_cloud.py'
cfg = Config.fromfile(cfg_file)

data_train = build_dataset(cfg.data.train)
# %%
# Inspect One sample
# idx=0
idx = random.randint(0,100)
img_info = data_train.img_infos[idx]
ann_info = data_train.get_ann_info(idx)
sample   = data_train[idx]
img_meta = sample['img_meta'].data
img 	 = sample['img'].data
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
# %%
mean = img_meta['img_norm_cfg']['mean']
std  = img_meta['img_norm_cfg']['std']

#retrieve input image
img_input = mmcv.rgb2bgr(img.permute(1,2,0).numpy()*std+mean)
result = (gt_labels.numpy(),gt_bboxes.numpy(),gt_masks)
filename = img_info['filename'].replace('.jpg','')
classes_name = ['background']+list(data_train.CLASSES)
img_out = show_gt(
			img_input, result, classes_name, show=False, out_file=f'test_{filename}.png')
# visualize_gt(img_input,gt_labels.numpy(),gt_bboxes.numpy(),gt_masks,classes_name)