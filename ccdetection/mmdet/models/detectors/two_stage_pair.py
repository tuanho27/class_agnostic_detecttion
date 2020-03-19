import torch
import torch.nn as nn
from time import time
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, delta2bbox, multiclass_nms
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from mmdet.ops import nms

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
        """
        Args:
            Since the dataset pipeline is changed to train pair images, but we don't want to change the 2 stage process in base detector
            Therefore we keep all the name of variables, but its values have beed changed 
            img: contain pair of 2 batch images that have at least 1 common class, 
                 shape [[batch,c,H,W],[batch,c,H,W]]
            img_meta: same img but content the meta infor. of images, similar for gt_bboxes, gt_labels

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        assert len(img) == 2 ## this variable always have len == 2

        ## forward each batch images in pair
        rpn_outputs = []
        for i in range(2):
            rpn_outputs.append(self.forward_train_single(img[i], img_meta[i],
                                                       gt_bboxes[i],
                                                       gt_labels[i],
                                                       gt_bboxes_ignore=None,
                                                       gt_masks=None,
                                                       proposals=None))

        losses_rpn = dict(rpn_losses_cls = rpn_outputs[0]['rpn_losses']['rpn_losses_cls'] + 
                                                        rpn_outputs[1]['rpn_losses']['rpn_losses_cls'],
                          rpn_losses_box = rpn_outputs[0]['rpn_losses']['rpn_losses_box'] + 
                                                        rpn_outputs[1]['rpn_losses']['rpn_losses_box'])
        losses.update(losses_rpn)

        ## calculate cost function for each pair in two batch images
        num_imgs = img[0].size()[0] 
        if self.with_bbox:
            loss_siameses = []
            loss_relations = []
            for i in range(num_imgs):
                ## Prepare positive pairs and negative pairs:
                foreground_index_0 = rpn_outputs[0]['proposal_label'][i].nonzero()
                foreground_index_1 = rpn_outputs[1]['proposal_label'][i].nonzero()
                pairs = []
                pairs_feats = []                
                pairs_targets = []

                if len(foreground_index_0) != 0 and len(foreground_index_1) != 0:
                    for idx0 in foreground_index_0:
                        for idx1 in foreground_index_1:
                            if rpn_outputs[0]['proposal_label'][i][idx0] == rpn_outputs[1]['proposal_label'][i][idx1]:
                                pairs.append(torch.cat(([rpn_outputs[0]['proposal_list'][i][idx0],
                                                                    rpn_outputs[1]['proposal_list'][i][idx1]]),dim=0))
                                pairs_feats.append(torch.cat(([rpn_outputs[0]['bbox_feats'][i][idx0],
                                                                       rpn_outputs[1]['bbox_feats'][i][idx1]]),dim=0))
                                pairs_targets.append(idx1>0) ## just keep tensor binary value

                            else:
                                pairs.append(torch.cat(([rpn_outputs[0]['proposal_list'][i][idx0],
                                                                    rpn_outputs[1]['proposal_list'][i][idx1]]),dim=0))

                                pairs_feats.append(torch.cat(([rpn_outputs[0]['bbox_feats'][i][idx0],
                                                                       rpn_outputs[1]['bbox_feats'][i][idx1]]),dim=0))
                                pairs_targets.append(idx1==0) ## just keep tensor binary value

                    if len(pairs) > 0:
                        pairs = torch.stack(pairs) 
                        pairs_feats = torch.stack(pairs_feats)  
                        pairs_targets = torch.cat(pairs_targets)       

                        ## Siamese matching loss
                        if self.siamese_matching_head is not None:
                            loss_siameses.append(self.siamese_matching_head(pairs, pairs_targets, pairs_feats)) 

                        ## relation matching loss
                        if self.relation_matching_head is not None:
                            loss_relations.append(self.relation_matching_head(pairs, pairs_targets, pairs_feats)) 

            loss_siamese =  dict(loss_siamese=torch.stack(loss_siameses).mean())  
            loss_relation =  dict(loss_relation=torch.stack(loss_relations).mean())

            losses.update(loss_siamese)
            losses.update(loss_relation)

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
            Same above but for single batch 
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
            proposal_list, proposal_label  = self.rpn_head.get_proposals_w_label(*proposal_inputs)

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
            bbox_feats = []
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
                rois = bbox2roi([sampling_result.bboxes])
                bbox_feats.append(self.bbox_roi_extractor(tuple(j[i:i+1] for j in x), rois))
                if self.with_shared_head:
                    bbox_feats.append(self.shared_head(bbox_feats))

            rpn_losses = dict(rpn_losses_cls=torch.stack(rpn_losses['loss_rpn_cls']).sum(),
                            rpn_losses_box =torch.stack(rpn_losses['loss_rpn_bbox']).sum())

        return  dict(rpn_losses=rpn_losses,
                     proposal_list=proposal_list,
                     proposal_label = proposal_label, 
                     bbox_feats=bbox_feats)

    ### For inference pair images (this function re-used as augmementation test)
    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        result_outputs = []
        pairs = []
        pairs_feats = []
        for i, im in enumerate(img):
            output = self.simple_test_single(im[0], img_meta[i][0])
            result_outputs.append(output)

        ## pairs number of proposal
        for i in range(result_outputs[0]['proposal_list'].size()[0]):
            for j in range(result_outputs[1]['proposal_list'].size()[0]): 
                pairs.append(torch.stack((result_outputs[0]['proposal_list'][i],
                                                    result_outputs[1]['proposal_list'][j]), dim=0))
                pairs_feats.append(torch.stack((result_outputs[0]['proposal_feats'][i],
                                                    result_outputs[1]['proposal_feats'][j]), dim=0))   
        pairs = torch.stack(pairs) 
        pairs_feats = torch.stack(pairs_feats)  

        pair_score_siamese = self.siamese_matching_head.forward_test(pairs, pairs_feats)
        pair_score_relation = self.relation_matching_head.forward_test(pairs, pairs_feats)
        print("Total pairs: ",pairs_feats.size()[0])
        return pairs[torch.argsort(pair_score_relation)[-30:]]


    def simple_test_single(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        proposal_list, _ =  nms(proposal_list[0],0.15)
        rois = bbox2roi([proposal_list])
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)

        return dict(proposal_list=proposal_list, proposal_feats=roi_feats)

### additional cls ###
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
