import torch
import torch.nn as nn
from time import time
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, delta2bbox, multiclass_nms
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class TwoStagePairDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 siamese_matching_head=None,
                 relation_matching_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStagePairDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)
        
        if siamese_matching_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.siamese_matching_head = builder.build_head(siamese_matching_head)

        if relation_matching_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.relation_matching_head = builder.build_head(relation_matching_head) 

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)


    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStagePairDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

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
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs


    ## add match function
    def common_member(self, a, b): 
        a_set = set(a.detach().cpu().numpy()) 
        b_set = set(b.detach().cpu().numpy()) 
        common_list = a_set & b_set
        if common_list: 
            return [*common_list,]
        else:
            return None


    #######################################################
    ## for train pair images
    def forward_train(self,img,
                        img_meta,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        proposals=None):
        losses = dict()
        rpn_outputs = []
        for i, im in enumerate(img):
            rpn_output = self.forward_train_single(im, img_meta[i],
                                                       gt_bboxes[i],
                                                       gt_labels[i],
                                                       gt_bboxes_ignore=None,
                                                       gt_masks=None,
                                                       proposals=None)
            rpn_outputs.append(rpn_output)
            losses.update(rpn_output['rpn_losses'])

        if self.with_bbox:
            ## Prepare positive pairs and negative pairs:
            foreground_index_0 = rpn_outputs[0]['proposal_label'].nonzero()
            foreground_index_1 = rpn_outputs[1]['proposal_label'].nonzero()

            pairs = []
            pairs_feats = []
            
            ## add targets from anchor head for refine RPN loss
            pairs_targets = []
            pairs_bbox_target = []
            pairs_bbox_target_weight = []

            if len(foreground_index_0) == 0 or len(foreground_index_1)==0:
                pass 
                # print("Length Pairs: ", 0)

            else:
                for idx0 in foreground_index_0:
                    for idx1 in foreground_index_1:
                        if rpn_outputs[0]['proposal_label'][idx0] == rpn_outputs[1]['proposal_label'][idx1]:
                            pairs.append(torch.cat(([rpn_outputs[0]['proposal_list'][idx0],
                                                                rpn_outputs[1]['proposal_list'][idx1]]),dim=0))

                            pairs_feats.append(torch.cat(([rpn_outputs[0]['bbox_feats'][idx0],
                                                                rpn_outputs[1]['bbox_feats'][idx1]]),dim=0))
                            pairs_targets.append(idx1>0)

                            pairs_bbox_target.append(torch.cat(([rpn_outputs[0]['proposal_bbox'][idx0],
                                                                rpn_outputs[1]['proposal_bbox'][idx1]]),dim=0))

                            pairs_bbox_target_weight.append(torch.cat(([rpn_outputs[0]['proposal_bbox_weight'][idx0],
                                                                rpn_outputs[1]['proposal_bbox_weight'][idx1]]),dim=0))
                        else:
                            pairs.append(torch.cat(([rpn_outputs[0]['proposal_list'][idx0],
                                                                rpn_outputs[1]['proposal_list'][idx1]]),dim=0))

                            pairs_feats.append(torch.cat(([rpn_outputs[0]['bbox_feats'][idx0],
                                                                rpn_outputs[1]['bbox_feats'][idx1]]),dim=0))
                            pairs_targets.append(idx1==0)

                            pairs_bbox_target.append(torch.cat(([rpn_outputs[0]['proposal_bbox'][idx0],
                                                                rpn_outputs[1]['proposal_bbox'][idx1]]),dim=0))

                            pairs_bbox_target_weight.append(torch.cat(([rpn_outputs[0]['proposal_bbox_weight'][idx0],
                                                                rpn_outputs[1]['proposal_bbox_weight'][idx1]]),dim=0))

                # print("Length Pairs: ", len(pairs))
                if len(pairs) > 0:
                    pairs = torch.stack(pairs) 
                    pairs_feats = torch.stack(pairs_feats)  
                    pairs_targets = torch.cat(pairs_targets)       
                    pairs_bbox_target = torch.stack(pairs_bbox_target)        
                    pairs_bbox_target_weight = torch.stack(pairs_bbox_target_weight)        

                    ## Siamese matching loss
                    if self.siamese_matching_head is not None:
                        loss_siamese = self.siamese_matching_head(pairs, pairs_targets, pairs_feats, 
                                                                    pairs_bbox_target, pairs_bbox_target_weight)
                        losses.update(loss_siamese)

                    ## relation matching loss
                    if self.relation_matching_head is not None:
                        loss_relation = self.relation_matching_head(pairs, pairs_targets, pairs_feats, 
                                                                    pairs_bbox_target, pairs_bbox_target_weight)
                        losses.update(loss_relation)

                else:
                    pass

        return losses


    def forward_train_single(self,img,
                            img_meta,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_ignore=None,
                            gt_masks=None,
                            proposals=None):
        """
        Args:

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        rpn_losses = dict()
        ## RPN forward and loss L(RPN) between RPN feature map by anchors and ground-truths
        ## Output candidate proposals 
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes,img_meta,
                                                self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

            proposal_cfg = self.train_cfg.get('rpn_proposal',self.train_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, self.train_cfg,  gt_bboxes, gt_labels)
            proposal_list, proposal_label, proposal_bbox, proposal_bbox_weight  = self.rpn_head.get_proposals_w_label(*proposal_inputs)

        else:
            proposal_list = proposals
    
        # Assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None: 
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
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
            
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        rpn_losses = dict(rpn_losses_cls=torch.stack(rpn_losses['loss_rpn_cls']).sum(),
                          rpn_losses_box =torch.stack(rpn_losses['loss_rpn_bbox']).sum())

        return  dict(rpn_losses=rpn_losses,
                     proposal_list=torch.cat(proposal_list),
                     proposal_label = torch.cat(proposal_label), 
                     proposal_bbox =torch.cat(proposal_bbox),
                     proposal_bbox_weight = torch.cat(proposal_bbox_weight),
                     bbox_feats=bbox_feats)


    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
