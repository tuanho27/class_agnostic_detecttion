import mmcv
import numpy as np
import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .fcos import FCOS
from .mask_single_stage import MaskSingleStateDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class FCOSMask(MaskSingleStateDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(FCOSMask, self).__init__(
            backbone,
            neck,
            bbox_head,
            mask_roi_extractor,
            mask_head,
            train_cfg,
            test_cfg,
            pretrained,
        )

        self.mask_roi_extractor = builder.build_roi_extractor(
            mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)

    # def __init__(self,
    #              backbone,
    #              neck,
    #              bbox_head,
    #              train_cfg=None,
    #              test_cfg=None,
    #              pretrained=None):
    #     super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
    #                                test_cfg, pretrained)

    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None,
    #                   gt_masks=None):
    #     self.timer.since_last_check()
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
    #     # losses = self.bbox_head.loss(
    #     #     *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    #     losses = {}
    #     # The below code is adopted from two_stage.py
    #     # Proposal bboxes by nms/get bbox with top predicted prob
    #     proposal_cfg = self.train_cfg.get('rpn_proposal',
    #                                           self.test_cfg)

    #     proposal_inputs = outs + (img_metas, proposal_cfg)
    #     bbox_results = self.bbox_head.get_bboxes(*proposal_inputs)
    #     #self.time_records['proposal'].append(self.timer.since_last_check())
    #     # collect
    #     bbox_targets = [(bb, lbl) for bb, lbl in zip(gt_bboxes, gt_labels)]
    #     proposal_list = [det_bboxes for det_bboxes, det_labels in bbox_results]

        
    #     # Sampling the Proposal to match with format of fcn_mask
    #     bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
    #     bbox_sampler  = build_sampler(self.train_cfg.rcnn.sampler, context=self)            
    #     num_imgs = img.size(0)
    #     if gt_bboxes_ignore is None:
    #         gt_bboxes_ignore = [None for _ in range(num_imgs)]

    #     sampling_results = []
    #     for i in range(num_imgs):
    #         ith_proposal = proposal_list[i]
    #         assign_result = bbox_assigner.assign(ith_proposal,
    #                                                 gt_bboxes[i],
    #                                                 gt_bboxes_ignore[i],
    #                                                 gt_labels[i])
    #         sampling_result = bbox_sampler.sample(
    #             assign_result,
    #             ith_proposal,
    #             gt_bboxes[i],
    #             gt_labels[i],
    #             feats=[lvl_feat[i][None] for lvl_feat in x])
    #         sampling_results.append(sampling_result)

    #     # #self.time_records['assign'].append(self.timer.since_last_check())
    #     # sampling_results[0].pos_bboxes = gt_bboxes#bbox2roi()

    #     # rois = bbox2roi(
    #     #             [res.pos_bboxes for res in sampling_results])
    #     rois = bbox2roi(gt_bboxes)

    #     # import ipdb; ipdb.set_trace()

    #     mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)

    #     # print('[Train], mask_feats:', mask_feats.shape)
    #     mask_pred = self.mask_head(mask_feats)

    #     mask_targets = self.mask_head.get_target(sampling_results,
    #                                             gt_masks,
    #                                             self.train_cfg.rcnn)
    #     # pos_labels = torch.cat(
    #     #         [res.pos_gt_labels for res in sampling_results])
    #     pos_labels = torch.cat(gt_labels)
    #     # print(gt_labels)
    #     loss_mask = self.mask_head.loss(mask_pred, mask_targets,
    #                                         pos_labels)
    #     losses.update(loss_mask)
    #     # self.mask_time.append(self.timer.since_last_check())
    #     # #self.time_records['mask'].append(self.timer.since_last_check())

    #     return losses