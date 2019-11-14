# from .registry import DATASETS
import numpy as np
import os.path as osp
import os
import torch

import cv2
import pickle

from .custom import CustomDataset
from .registry import DATASETS

from .create_cloud_dataset import rle2mask

@DATASETS.register_module
class CloudDataset(CustomDataset):

    CLASSES = ('Fish', 'Flower', 'Gravel', 'Sugar')

    def __init__(self, preload_mask=True, with_ignore_bboxes=True, **kwargs):
        super(CloudDataset, self).__init__(**kwargs)
        self.preload_mask = preload_mask
        self.with_ignore_bboxes=with_ignore_bboxes

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as handle:
            data_infos = pickle.load(handle)
        img_infos=data_infos['img_infos']
        self.ann_infos = data_infos['ann_infos']
        return img_infos

    def get_ann_info(self, idx):
        img_info = self.img_infos[idx]
        # We convert rle from ann into real mask
        h,w= img_info['height'],img_info['width']
        ann_info = self.ann_infos[idx]
        fann_info = ann_info.copy()
        if self.preload_mask:
            masks = [rle2mask(m,(h,w)) for m in ann_info['masks']]
            masks = [m[np.newaxis,:] for m in masks]
            masks = np.concatenate(tuple(masks),axis=0)
            # formatted output anno 
            fann_info['masks']=masks
        else:
            del fann_info['masks']
        if self.with_ignore_bboxes:
            # Add ignore mask to avoid error
            fann_info['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
        return fann_info