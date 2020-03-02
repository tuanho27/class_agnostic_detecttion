import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from .embedding_matching import RelationMatching, SiameseMatching


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
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
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        
        self.siamese_matching = SiameseMatching()
        self.relation_matching = RelationMatching()

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
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

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
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

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
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
    def forward_pair_train(self,img,
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

        x0 = self.extract_feat(img[0])
        x1 = self.extract_feat(img[1])
        losses = dict()

        ## RPN forward and loss L(RPN) between RPN feature map by anchors and ground-truths
        ## Output candidate proposals 
        if self.with_rpn:
            ## for set image 0
            rpn_outs_img0 = self.rpn_head(x0)
            rpn_loss_inputs_img0 = rpn_outs_img0 + (gt_bboxes[0], img_meta[0],
                                          self.train_cfg.rpn)
            rpn_losses_img0 = self.rpn_head.loss(
                *rpn_loss_inputs_img0, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses_img0)

            proposal_cfg = self.train_cfg.get('rpn_proposal',self.test_cfg.rpn)
            proposal_inputs_img0 = rpn_outs_img0 + (img_meta[0], proposal_cfg)
            proposal_list_img0 = self.rpn_head.get_bboxes(*proposal_inputs_img0)

            ## for set image 1
            rpn_outs_img1 = self.rpn_head(x1)
            rpn_loss_inputs_img1 = rpn_outs_img1 + (gt_bboxes[1], img_meta[1],
                                          self.train_cfg.rpn)
            rpn_losses_img1 = self.rpn_head.loss(
                *rpn_loss_inputs_img1, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses_img1)

            proposal_cfg = self.train_cfg.get('rpn_proposal',self.test_cfg.rpn)
            proposal_inputs_img1 = rpn_outs_img1 + (img_meta[1], proposal_cfg)
            proposal_list_img1 = self.rpn_head.get_bboxes(*proposal_inputs_img1)

        else:
            proposal_list = proposals
        
        # Assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            ## for set image 0
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img[0].size(0)
            if gt_bboxes_ignore is None: 
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results_img0 = []
            for i in range(num_imgs):
                assign_result_img0 = bbox_assigner.assign(proposal_list_img0[i],
                                                     gt_bboxes[0][i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[0][i])
                sampling_result_img0 = bbox_sampler.sample(
                    assign_result_img0,
                    proposal_list_img0[i],
                    gt_bboxes[0][i],
                    gt_labels[0][i],
                    feats=[lvl_feat[i][None] for lvl_feat in x0])
                sampling_results_img0.append(sampling_result_img0)

            ## for set image 1
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img[1].size(0)
            if gt_bboxes_ignore is None: 
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results_img1 = []
            for i in range(num_imgs):
                assign_result_img1 = bbox_assigner.assign(proposal_list_img1[i],
                                                     gt_bboxes[1][i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[1][i])
                sampling_result_img1 = bbox_sampler.sample(
                    assign_result_img1,
                    proposal_list_img1[i],
                    gt_bboxes[1][i],
                    gt_labels[1][i],
                    feats=[lvl_feat[i][None] for lvl_feat in x1])
                sampling_results_img1.append(sampling_result_img1)
            

        ## Prepare positive proposal (P* (P1 & P2)) w/ labels after get proposals from RPN 
        rois_img0 = bbox2roi([res.bboxes for res in sampling_results_img0])
        bbox_feats_img0 = self.bbox_roi_extractor(
            x0[:self.bbox_roi_extractor.num_inputs], rois_img0)
        if self.with_shared_head:
            bbox_feats_img0 = self.shared_head(bbox_feats_img0)
        cls_score_img0, bbox_pred_img0 = self.bbox_head(bbox_feats_img0)
        post_proposal_img0, post_proposal_label_img0 = self.get_post_proposals(rois_img0, 
                                                        bbox_feats_img0, img_meta[0])
        

        rois_img1 = bbox2roi([res.bboxes for res in sampling_results_img1])
        bbox_feats_img1 = self.bbox_roi_extractor(
            x1[:self.bbox_roi_extractor.num_inputs], rois_img1)
        if self.with_shared_head:
            bbox_feats_img1 = self.shared_head(bbox_feats_img1)
        cls_score_img1, bbox_pred_img1 = self.bbox_head(bbox_feats_img1)
        post_proposal_img1, post_proposal_label_img1 = self.get_post_proposals(rois_img1, 
                                                        bbox_feats_img1, img_meta[1])

        ## Prepare positive pairs and negative pairs:
        if self.with_bbox:
            pairs_positive = []
            pairs_negative = []

            common_list = self.common_member(post_proposal_label_img0, post_proposal_label_img1) 
            if common_list is not None:
                ## list positive pairs and negative pairs if two images have the same category
                for i in range(0,len(post_proposal_img0)):
                    for j in range(0,len(post_proposal_img1)):
                        if post_proposal_label_img0[i] == post_proposal_label_img1[j]:
                            pairs_positive.append(torch.stack(([post_proposal_img0[i],post_proposal_img1[j]]),dim=0))
                        else:
                            pairs_negative.append(torch.stack(([post_proposal_img0[i],post_proposal_img1[j]]),dim=0))

            ## negative pairs for unmatching any category
            else:
                for i in range(0,len(post_proposal_img0)):
                    for j in range(0,len(post_proposal_img1)):
                        pairs_negative.append(torch.stack(([post_proposal_img0[i],post_proposal_img1[j]]),dim=0))

        ## Calculate L(mat)
        try:
            pairs_positive = torch.stack(pairs_positive,dim=0)
        except:
            import ipdb; ipdb.set_trace()
        pairs_negative = torch.stack(pairs_negative,dim=0)

        ## positive proposal feats
        pairs_positive_rois_0 = bbox2roi([pairs_positive[:,:1,:4].view(-1,4)])
        pairs_positive_feats_0 = self.bbox_roi_extractor(
                            x0[:self.bbox_roi_extractor.num_inputs], pairs_positive_rois_0)

        pairs_positive_rois_1 = bbox2roi([pairs_positive[:,1:2,:4].view(-1,4)])
        pairs_positive_feats_1 = self.bbox_roi_extractor(
                            x1[:self.bbox_roi_extractor.num_inputs], pairs_positive_rois_1)

        ## negative proposal feats
        pairs_negative_rois_0 = bbox2roi([pairs_negative[:,:1,:4].view(-1,4)])
        pairs_negative_feats_0 = self.bbox_roi_extractor(
                            x0[:self.bbox_roi_extractor.num_inputs], pairs_negative_rois_0)

        pairs_negative_rois_1 = bbox2roi([pairs_negative[:,1:2,:4].view(-1,4)])
        pairs_negative_feats_1 = self.bbox_roi_extractor(
                            x1[:self.bbox_roi_extractor.num_inputs], pairs_negative_rois_1)

        # Siamese matching loss
        # loss_siamese = self.siamese_matching(pairs_positive, pairs_positive_feats_0,pairs_positive_feats_1,
                                            #  pairs_negative, pairs_negative_feats_0,pairs_negative_feats_1)

        # losses.update(loss_siamese)
        # relation matching loss
        loss_relation = self.relation_matching(pairs_positive, pairs_positive_feats_0, pairs_positive_feats_1,
                                             pairs_negative, pairs_negative_feats_0,pairs_negative_feats_1)
        losses.update(loss_relation)
        ## Keep the calculate L(ref) for candidate positive proposals
        ## TODO: a more flexible way to decide which feature maps to use
        # set images 0
        bbox_targets_img0 = self.bbox_head.get_target(sampling_results_img0,
                                                    gt_bboxes[0], gt_labels[0],
                                                    self.train_cfg.rcnn)
        loss_bbox_img0 = self.bbox_head.loss(cls_score_img0, bbox_pred_img0,*bbox_targets_img0)
        losses.update(loss_bbox_img0)

        # set images 1
        bbox_targets_img1 = self.bbox_head.get_target(sampling_results_img1,
                                                    gt_bboxes[1], gt_labels[1],
                                                    self.train_cfg.rcnn)
        loss_bbox_img1 = self.bbox_head.loss(cls_score_img1, bbox_pred_img1,*bbox_targets_img1)
        losses.update(loss_bbox_img1)

        ## Mask head forward and loss
        # if self.with_mask:
        #     if not self.share_roi_extractor:
        #         pos_rois = bbox2roi(
        #             [res.pos_bboxes for res in sampling_results])
        #         mask_feats = self.mask_roi_extractor(
        #             x[:self.mask_roi_extractor.num_inputs], pos_rois)
        #         if self.with_shared_head:
        #             mask_feats = self.shared_head(mask_feats)
        #     else:
        #         pos_inds = []
        #         device = bbox_feats.device
        #         for res in sampling_results:
        #             pos_inds.append(
        #                 torch.ones(
        #                     res.pos_bboxes.shape[0],
        #                     device=device,
        #                     dtype=torch.uint8))
        #             pos_inds.append(
        #                 torch.zeros(
        #                     res.neg_bboxes.shape[0],
        #                     device=device,
        #                     dtype=torch.uint8))
        #         pos_inds = torch.cat(pos_inds)
        #         mask_feats = bbox_feats[pos_inds]
        #     mask_pred = self.mask_head(mask_feats)

        #     mask_targets = self.mask_head.get_target(sampling_results,
        #                                              gt_masks,
        #                                              self.train_cfg.rcnn)
        #     pos_labels = torch.cat(
        #         [res.pos_gt_labels for res in sampling_results])
        #     loss_mask = self.mask_head.loss(mask_pred, mask_targets,
        #                                     pos_labels)
        #     losses.update(loss_mask)

        return losses


    def get_post_proposals(self, rois, roi_feats, img_meta, rescale=False):
        """This function supports to get final bbox base on rpn proposals.
           This step usually be used in test phase, but re-used in training to the purpose of 
           preparing positive proposal pair as paper https://ieeexplore.ieee.org/document/8606132
        """
        assert self.with_bbox, "Bbox head must be implemented."
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        rcnn_cfg = dotdict(dict(score_thr=0.0, 
                             nms=dict(type='nms', iou_thr=0.0), 
                             max_per_img=self.train_cfg.rpn_proposal.num_post_proposal))
        proposal_bboxes, proposal_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_cfg)
        return proposal_bboxes, proposal_labels


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
