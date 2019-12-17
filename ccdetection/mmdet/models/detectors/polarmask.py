from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn
from mmdet.core import bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
import time
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.core import auto_fp16, force_fp32, mask_target

@DETECTORS.register_module
class PolarMask(SingleStageDetector):

	def __init__(self,
				 backbone,
				 neck,
				 bbox_head,
				 semseg_head=None,
				 yolact_proto_head=None,
				 train_cfg=None,
				 test_cfg=None,
				 pretrained=None):
				# loss_cls_combine=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)):

		super(PolarMask, self).__init__(backbone, neck, bbox_head, train_cfg, 
								   test_cfg, pretrained)
		if semseg_head is not None:
			self.semseg_head = builder.build_head(semseg_head)
			self.semseg_head.init_weights()

		if yolact_proto_head is not None:
			self.yolact_proto_head = builder.build_head(yolact_proto_head)
			self.yolact_proto_head.init_weights()

		self.init_weights(pretrained=pretrained)

	@property
	def with_semseg(self):
		return hasattr(self, 'semseg_head') and self.semseg_head is not None
	
	@property
	def with_yolact(self):
		return hasattr(self, 'yolact_proto_head') and self.yolact_proto_head is not None

	def init_weights(self, pretrained=None):
		super(PolarMask, self).init_weights(pretrained)
		self.backbone.init_weights(pretrained=pretrained)

		if self.with_neck:
			if isinstance(self.neck, nn.Sequential):
				for m in self.neck:
					m.init_weights()
			else:
				self.neck.init_weights()

		self.bbox_head.init_weights()

	# @force_fp32(apply_to=('logit_4D','mask_4D'))
	def forward_train(self,
					  img,
					  img_metas,
					  gt_bboxes,
					  gt_labels,
					  gt_masks=None,
					  gt_bboxes_ignore=None,
					  gt_fg_mask=None,
					  _gt_labels=None,
					  _gt_bboxes=None,
					  _gt_masks=None,
					  ):

		if _gt_labels is not None:
			extra_data = dict(_gt_labels=_gt_labels,
							  _gt_bboxes=_gt_bboxes,
							  _gt_masks=_gt_masks)
		else:
			extra_data = None

		x = self.extract_feat(img)
		outs = self.bbox_head(x)
		loss_inputs = outs[:][:4] + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

		losses = self.bbox_head.loss(
			*loss_inputs,
			gt_masks = gt_masks,
			gt_bboxes_ignore=gt_bboxes_ignore,
			extra_data=extra_data
		)

		if self.with_semseg:
			mask_pred, new_outs_cls = self.semseg_head(x, outs)
			# loss_semseg, loss_combine_cls = self.semseg_head.loss(mask_pred, 
			# 														gt_fg_mask, 
			# 														new_outs_cls, 
			# 														outs[1], 
			# 														extra_data)
			loss_semseg = self.semseg_head.loss(mask_pred, 
												gt_fg_mask, 
												new_outs_cls, 
												outs[1], 
												extra_data) 
			losses.update(loss_semseg)
			# losses.update(loss_combine_cls)
		# import ipdb; ipdb.set_trace()
		if self.with_yolact:			
			protonet_coff_new = [] 
			for out in outs[:][4]:
				if out.shape[2] != outs[:][4][0].shape[2]:
					out = F.interpolate(out, size=(outs[:][4][0].shape[2],outs[:][4][0].shape[3]), mode="bilinear")
				protonet_coff_new.append(out)
			protonet_coff = torch.cat(protonet_coff_new, dim=1)

			proto_mask, protonet_coff_new = self.yolact_proto_head(x, protonet_coff)
			loss_proto_mask = self.yolact_proto_head.loss(gt_fg_mask, 
														  proto_mask,
														  protonet_coff, 
														  outs[:][1],
														  outs[:][2],
														  extra_data)

			## add mask 4D for rmi loss
			# proto_mask, protonet_coff_new, logit_4D = self.yolact_proto_head(x, protonet_coff, img)
			# mask_4D = []
			# for gt_mask, gt_label in zip(gt_masks, gt_labels):
			# 	for mask, label in zip(gt_mask, gt_label):
			# 		mask[mask>0] = label.cpu()
			# 	mask = np.sum(gt_mask,axis=0)
			# 	mask_4D.append(mask)
			# mask_4D = torch.from_numpy(np.array(mask_4D).astype(np.long)).cuda()
			# loss_proto_mask, loss_rmi = self.yolact_proto_head.loss(gt_fg_mask, 
			# 											  proto_mask,
			# 											  protonet_coff_new, 
			# 											  logit_4D, 
			# 											  mask_4D)

			## update loss
			losses.update(loss_proto_mask)
			# losses.update(loss_rmi)

		return losses

	def simple_test(self, img, img_meta, rescale=False):
		x = self.extract_feat(img)
		outs = self.bbox_head(x)
		bbox_inputs = outs[:4] + (img_meta, self.test_cfg, rescale)

		bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
		import ipdb; ipdb.set_trace()
		results = [
			bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
			for det_bboxes, det_labels, det_masks in bbox_list]

		bbox_results = results[0][0]
		mask_results = results[0][1]
		return bbox_results, mask_results


@DETECTORS.register_module
class PolarMaskONNX(PolarMask):
        """
                This class is to support exporting into ONNX file, by removing img_me
        """
        def forward(self, imgs):
                x = self.extract_feat(imgs)
                outs = self.bbox_head(x)
                return outs