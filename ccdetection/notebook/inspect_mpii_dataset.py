# %%
from mmdet.datasets import MPIIDataset
import mmcv
import matplotlib.pyplot as plt
import numpy as np
# from mmdet.datasets.visualize_ground_truth import show_gt, visualize_gt

data_root='/home/cybercore/Workspace/dataset/mpii/'

img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadPoseAnnotations', with_joints=True, with_heatmap=True),
	dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
    dict(type='Collect',
        keys=['img', 'gt_joints', 'gt_heatmap'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

data_train=MPIIDataset(ann_file=data_root + 'annot/valid_multi.json',
						img_prefix=data_root + 'images/',
						pipeline=train_pipeline)


# %%
# Inspect One sample
idx = 0
img_info = data_train.img_infos[idx]
ann_info = data_train.get_ann_info(idx)
sample = data_train[idx]

# img_meta = sample['img_meta'].data
# img = sample['img'].data
# gt_bboxes= sample['gt_bboxes'].data
# gt_labels= sample['gt_labels'].data
# gt_masks = sample['gt_masks'].data
import pdb; pdb.set_trace()
# # %%
# mean = img_meta['img_norm_cfg']['mean']
# std = img_meta['img_norm_cfg']['std']
# #retrieve inpyt image
# img_input = mmcv.rgb2bgr(img.permute(1,2,0).numpy()*std+mean)
# result = (gt_labels.numpy(),gt_bboxes.numpy(),gt_masks)
# filename = img_info['id']
# classes_name = ['background']+list(data_train.CLASSES)
# img_out = show_gt(
# 			img_input, result, classes_name, show=False, out_file=f'test_{filename}.png')
# visualize_gt(img_input,gt_labels.numpy(),gt_bboxes.numpy(),gt_masks,classes_name)

