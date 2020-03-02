import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset
import random
from .pipelines import Compose
from .registry import DATASETS
# from pyson.utils import memoize
from instaboost import get_new_data, InstaBoostConfig

@DATASETS.register_module
class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False, num_samples=None, instaboost=False):
        # import ipdb; ipdb.set_trace()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.instaboost = instaboost
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        timer = mmcv.Timer()
        self.img_infos = self.load_annotations(self.ann_file)
        print("Loaded annotation times:", timer.since_start())
        if num_samples is not None:
            self.img_infos = self.img_infos[:num_samples]

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.prepare_train_img(0)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
            
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
             ## adding instaboost augmentation before start data pipeline 
        # if self.instaboost:
        #     img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        #     img_id = self.img_infos[idx]['id']
        #     ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        #     ann_info = self.coco.loadAnns(ann_ids)
        #     aug_flag = np.random.choice([0,1],p=[0.5,0.5])
        #     if aug_flag:
        #         ann_info, img = get_new_data(ann_info, img, None, background=None)
        #     ann_info = self._parse_ann_info(img_info, ann_info, True)

        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        # ### For visualize data after augmentation
        # from mmdet.apis.inference import show_result
        # from .coco import CocoDataset
        # import cv2
        
        # results = self.pipeline(results)
        # print(results.keys())
        # out_file = './work_dirs/test_dataset.jpg'
        # # img = results['img']
        # img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # bboxes = results['gt_bboxes']
        # score = np.ones(18)
        # bboxes = np.concatenate((bboxes,score.reshape(1,18).T),axis=1)
        # masks = results['gt_masks']
        # gt_result = bboxes, masks
        # # cv2.imwrite(out_file, img)
        # # image = cv2.add(img, masks)
        # cv2.imwrite(out_file, masks)
        # import ipdb; ipdb.set_trace()
        # show_result(img, gt_result, results['gt_labels'], CocoDataset.CLASSES,show=False, out_file=out_file)
        # return 0
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)



@DATASETS.register_module
class CustomPairDataset(Dataset):
    CLASSES = None
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False, num_samples=None, instaboost=False):
        # import ipdb; ipdb.set_trace()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.instaboost = instaboost
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        timer = mmcv.Timer()
        self.img_infos = self.load_annotations(self.ann_file)
        print("Dataset Length: ",len(self.img_infos))
        print("Loaded annotation times:", timer.since_start())
        if num_samples is not None:
            self.img_infos = self.img_infos[:num_samples]

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.prepare_train_img(0)
        # import ipdb; ipdb.set_trace()

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def common_member(self, a, b): 
        a_set = set(a) 
        b_set = set(b) 
        if (a_set & b_set): 
            return True 
        else: 
            return False

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
            
    def prepare_train_img(self, idx):
        CLASSES_IGNORE = ['chair', 'cow', 'diningtable', 'dog', 'horse']
        idx0 = idx
        img0_info = self.img_infos[idx0]
        ann0_info = self.get_ann_info(idx0)
        # if ann0_info['labels'] in CLASSES_IGNORE:
        for i in range(1,len(self.img_infos)):
            idx1 = int((i + random.randint(1,len(self.img_infos)))/2) 
            if idx1 != idx:
                img1_info = self.img_infos[idx1]
                ann1_info = self.get_ann_info(idx1)
                if self.common_member(ann0_info['labels'], ann1_info['labels']):
                    break

        results_0 = dict(img_info=img0_info, ann_info=ann0_info)
        results_1 = dict(img_info=img1_info, ann_info=ann1_info)

        if self.proposals is not None:
            results_0['proposals'] = self.proposals[idx0]
            results_1['proposals'] = self.proposals[idx1]
        self.pre_pipeline(results_0)
        self.pre_pipeline(results_1)

        data0 = self.pipeline(results_0)
        data1 = self.pipeline(results_1)
        
        data = dict(img_meta=[data0['img_meta'],data1['img_meta']],
                    img=[data0['img'],data1['img']],
                    gt_bboxes=[data0['gt_bboxes'],data1['gt_bboxes']],
                    gt_labels=[data0['gt_labels'],data1['gt_labels']],
                    )
        return data

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
