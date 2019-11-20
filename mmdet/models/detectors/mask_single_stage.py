import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class MaskSingleStateDetector(BaseDetector):
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
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MaskSingleStateDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.bbox_head = builder.build_head(bbox_head)

        self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(MaskSingleStateDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
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

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        
        proposals=outs
        rois = bbox2roi([proposals])
        mask_rois = rois[:100]
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], mask_rois)
        
        mask_pred = self.mask_head(mask_feats)
        outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        # import ipdb; ipdb.set_trace()
        # BBox head
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # The below code is adopted from two_stage.py
        # Proposal bboxes by nms/get bbox with top predicted prob
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg)
        # import ipdb; ipdb.set_trace()
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
        import ipdb; ipdb.set_trace()
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                    gt_bboxes[i],
                                                    gt_bboxes_ignore[i],
                                                    gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # Mask head
        pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
        mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
        mask_pred = self.mask_head(mask_feats)

        mask_targets = self.mask_head.get_target(sampling_results,
                                                gt_masks,
                                                self.train_cfg.rcnn)
        pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
        losses.update(loss_mask)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
