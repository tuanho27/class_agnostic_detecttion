import numpy as np
import torch
import torch.nn as nn
import mmcv
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
import torch.nn.functional as F

@DETECTORS.register_module
class MaskSingleStateDetector(BaseDetector, MaskTestMixin):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 semseg_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.timer = mmcv.Timer()
        #self.time_records = {x:[] for x in ['proposal', 'box', 'mask', 'assign']}

        super(MaskSingleStateDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.bbox_head = builder.build_head(bbox_head)

        self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)
        if (train_cfg is not None) and  (not 'train_mask' in train_cfg):
            train_cfg['train_mask'] = True

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)  

    def init_weights(self, pretrained=None):
        super(MaskSingleStateDetector, self).init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        
        self.bbox_head.init_weights()
        self.mask_roi_extractor.init_weights()
        self.mask_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img, mask_size=[100, 256, 14, 14]):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        """Batchsize must be 1!
        """
        # BBox
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_feats = torch.randn(*mask_size).cuda()
        outs_mask = self.mask_head(mask_feats)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        self.timer.since_last_check()
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if self.train_cfg.train_mask:   
            # The below code is adopted from two_stage.py
            # Proposal bboxes by nms/get bbox with top predicted prob
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg)
            proposal_inputs = outs + (img_metas, proposal_cfg)
            bbox_results = self.bbox_head.get_bboxes(*proposal_inputs)
            proposal_list = [det_bboxes for det_bboxes, det_labels in bbox_results]

            # Sampling the Proposal to match with format of fcn_mask
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler  = build_sampler(self.train_cfg.rcnn.sampler, context=self)            
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            sampling_results = []
            for i in range(num_imgs):
                ith_proposal = proposal_list[i]
                ith_proposal = gt_bboxes[i]

                assign_result = bbox_assigner.assign(ith_proposal, gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result, ith_proposal, gt_bboxes[i], gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)

            mask_targets = self.mask_head.get_target(sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

            rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)

            mask_targets = self.mask_head.get_target(sampling_results,gt_masks,self.train_cfg.rcnn)
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])


            # mask_pred = self.mask_head(mask_feats)       
            ### add manyfold mix-up 
            if pos_labels.shape[0]>0:
                mask_pred, out_cls, target_reweighted  = self.mask_head(mask_feats, pos_labels, manyfold_mixup=True, mixup_hidden=True)
                loss_cls_extra = {'loss_cls_extra': F.binary_cross_entropy_with_logits(out_cls, target_reweighted).view(1)}
                losses.update(loss_cls_extra)
            else:
                mask_pred = self.mask_head(mask_feats, pos_labels, mix_up=False, manyfold_mixup=False, mixup_hidden=True)

            loss_mask = self.mask_head.loss(mask_pred, mask_targets,pos_labels)
            losses.update(loss_mask)

            if self.with_semseg:
                gt_fg_mask = [ mask.sum(axis=0).clip(0,1)[None,None,...].astype('int64')
                                for mask in gt_masks ]
                gt_fg_mask = torch.from_numpy(np.concatenate(gt_fg_mask)).cuda()
                mask_pred, new_outs_cls = self.semseg_head(x, outs)
                loss_semseg = self.semseg_head.loss(mask_pred, 
                                                    gt_fg_mask, 
                                                    new_outs_cls) 
                losses.update(loss_semseg)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        """Batchsize must be 1!
        """
        # BBox
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        bbox_results = []
        for det_bboxes, det_labels in bbox_list:
            bb_result = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            bbox_results.append(bb_result)

        if not self.with_mask:
            return bbox_results[0]
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results[0], segm_results

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        # import ipdb; ipdb.set_trace()
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            if rescale:
                _bboxes = det_bboxes[:, :4] * scale_factor
            else :
                _bboxes = det_bboxes

            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)

            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
            
        return segm_result

    def get_seg_masks(self, mask_pred, ori_shape, scale_factor, rescale, threshold=0.5):
        if rescale:
            mask_pred = F.interpolate(mask_pred, size=ori_shape[:2], mode='bilinear', align_corners=True)

        mask_pred = mask_pred.sigmoid()
        mask_pred = (mask_pred > threshold).float()
        return mask_pred

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
