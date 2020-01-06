import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.cnn import normal_init
from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule

INF = 1e8

@HEADS.register_module
class YolactSemsegHead(nn.Module):
	def __init__(self,
				num_convs=4,
				num_convs_post=1,
				in_channels=256,
				conv_kernel_size=3,
				conv_out_channels=32,
				input_index=0,
				upsample_method='bilinear',
				upsample_ratio=2,
				# num_classes_semantic=1,
				conv_cfg=None,
				norm_cfg=None,
				loss_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
				loss_protonet_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
				loss_rmi = dict(type='RMILoss', num_classes=81, loss_weight_lambda=0.5, lambda_way=1),
				strides=(4, 8, 16, 32, 64),
				regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)),
				):
			
		super(YolactSemsegHead, self).__init__()	
		self.num_convs = num_convs
		self.num_convs_post = num_convs_post
		self.in_channels = in_channels
		self.conv_kernel_size = conv_kernel_size
		self.conv_out_channels = conv_out_channels
		self.input_index = input_index
		self.upsample_method = upsample_method
		self.upsample_ratio = upsample_ratio
		# self.num_classes_semantic = num_classes_semantic
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.fp16_enabled = False
		self.convs = nn.ModuleList()
		self.loss_mask = build_loss(loss_mask)
		self.loss_protonet_mask = build_loss(loss_protonet_mask)
		self.loss_rmi = build_loss(loss_rmi)
		self.strides = strides
		self.regress_ranges = regress_ranges

		for i in range(self.num_convs):
			padding = (self.conv_kernel_size - 1) // 2
			self.convs.append(
				ConvModule(
					self.in_channels,
					self.in_channels,
					self.conv_kernel_size,
					padding=padding,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg))
		
		upsample_in_channels = self.in_channels

		if self.upsample_method == 'deconv':
			self.upsample = nn.ConvTranspose2d(
									upsample_in_channels,
									self.in_channels,
									self.upsample_ratio,
									stride=self.upsample_ratio)
		else:
			self.upsample = None # using F.interpolate directly in forward func. 

		self.convs_post = nn.ModuleList()
		for i in range(self.num_convs_post):
			padding = (self.conv_kernel_size - 1) // 2
			self.convs_post.append(
				ConvModule(
					self.in_channels,
					self.in_channels,
					self.conv_kernel_size,
					padding=padding,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg))

		self.conv1x1 = ConvModule(
					self.in_channels,
					self.conv_out_channels,
					1,
					padding=0,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg)
		self.last_conv = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=1)

	def init_weights(self):
		for m in [self.upsample, self.convs, self.conv1x1, self.convs_post]:
			if m is None:
				continue
			if hasattr(m, 'weight'):
				nn.init.kaiming_normal_(
					m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

				nn.init.constant_(m.bias, 0)

	@auto_fp16()
	def forward(self, feats):
		x = feats[self.input_index]
		for conv in self.convs:
			x = conv(x)
		if self.upsample_method == 'deconv':
			x = self.upsample(x)
			x = self.relu(x)
		else:
			size = feats[self.input_index].size()[2:]
			x = F.interpolate(x, 
							size=(int(size[0]*2),int(size[1]*2)),
							mode=self.upsample_method,
							align_corners=True)
		
		for conv in self.convs_post:
			x = conv(x)

		## for protonet mask
		protonet_mask = self.conv1x1(x)
		## for semantic segmentation
		semseg_mask = self.last_conv(x)
		## for rmi loss
		size_logit=x.size()[2:]
		logit_4D = F.interpolate(self.last_conv(x), size=(int(size_logit[0]*2),int(size_logit[1]*2)), 
																mode='bilinear', align_corners=True)

		return protonet_mask, semseg_mask, logit_4D


	@force_fp32(apply_to=('protonet_masks','pos_mask_coefs','semseg_mask','mask_targets'))
	def loss(self, protonet_masks, pos_mask_coefs, semseg_mask, mask_targets, logit_4D, mask_target_4D,
				pos_bbox_pred, pos_bbox_target, gt_bboxes, img_metas):

		pos_mask_coefs = torch.tanh(pos_mask_coefs)
		protonet_masks_assembly = torch.sigmoid(torch.matmul(protonet_masks.permute(0,2,3,1), pos_mask_coefs.t()).permute(3,0,1,2))
		import ipdb; ipdb.set_trace()
		# for mask_p in  protonet_masks_assembly
		# protonet_masks = protonet_masks.permute(0,2,3,1).reshape(-1, 32)
		# protonet_masks_assembly = torch.sigmoid(torch.matmul(protonet_masks, pos_mask_coefs.t()))
		# import ipdb; ipdb.set_trace()
		# # protonet_masks_assembly = F.interpolate(protonet_masks_assembly.permute(0,3,1,2), 
		# 										# size=(768,1280),mode='bilinear', align_corners=True)
		# # mask_targets_resize = F.interpolate(mask_targets,
		# 										# size=(768,1280),mode='bilinear', align_corners=True)

		# for mask_p, mask_t, img_meta in zip(protonet_masks_assembly, mask_targets, img_metas):
		# 	for box in pos_bbox_target:
		# 		# box = box*torch.from_numpy(img_meta['scale_factor']).cuda()
		# 		box = box/4
		# 		mask_crop_t = self.crop_feat(mask_t[0], box.int())
		# 		print("Mask t: ",mask_crop_t)
		# 		for i, mask in enumerate(mask_p):
		# 			mask_crop_p = self.crop_feat(mask, box.int())
		# 			import ipdb; ipdb.set_trace()
		# 			loss_ins = self.loss_protonet_mask(mask_crop_p,mask_crop_t.reshape(-1).long())
		# 			print("Mask P: ",mask_crop_p)
		# 			print(loss_ins)
		# 			if(i==10):
		# 				import ipdb; ipdb.set_trace()
		# loss_protonet_mask = loss_

		## loss semseg mask
		num_classes_semantic = semseg_mask.shape[1]
		mask_pred = semseg_mask.permute(0, 2, 3, 1).reshape(-1, num_classes_semantic)
		mask_targets = mask_targets.permute(0, 2, 3, 1).reshape(-1, num_classes_semantic)
		loss_semseg_mask = self.loss_mask(mask_pred, mask_targets.long())

		## loss rmi
		# loss_rmi = self.loss_rmi(logit_4D, mask_target_4D)

		return dict(loss_protonet_mask=loss_protonet_mask,
					loss_semseg_mask=loss_semseg_mask)
					# loss_rmi=loss_rmi)


	def sanitize_coordinates(self,_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
		"""
		Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
		Also converts from relative to absolute coordinates and casts the results to long tensors.
		If cast is false, the result won't be cast to longs.
		Warning: this does things in-place behind the scenes so copy if necessary.
		"""
		_x1 = _x1 * img_size
		_x2 = _x2 * img_size
		if cast:
			_x1 = _x1.long()
			_x2 = _x2.long()
		x1 = torch.min(_x1, _x2)
		x2 = torch.max(_x1, _x2)
		x1 = torch.clamp(x1-padding, min=0)
		x2 = torch.clamp(x2+padding, max=img_size)

		return x1, x2


	def crop_feat(self, masks, bboxes, padding=1):
		# n, h, w = masks.size()
		# x1, x2 = self.sanitize_coordinates(bboxes[:, 0], bboxes[:, 2], w, padding, cast=True)
		# y1, y2 = self.sanitize_coordinates(bboxes[:, 1], bboxes[:, 3], h, padding, cast=True)
		# rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
		# cols = torch.arange(h, device=masks.device, dtype=y1.dtype).view(1, -1, 1).expand(n, h, w)
		h,w = masks.size()
		x1, x2 = bboxes[0], bboxes[2]
		y1, y2 = bboxes[1], bboxes[3]

		# rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1).expand(h, w)
		# cols = torch.arange(h, device=masks.device, dtype=y1.dtype).view(-1, 1).expand(h, w)

		# masks_left = rows >= x1.view(-1, 1)
		# masks_right = rows < x2.view(-1, 1)
		# masks_up = cols >= y1.view(-1, 1)
		# masks_down = cols < y2.view(-1, 1)

		# crop_mask = masks_left * masks_right * masks_up * masks_down
		crop_mask = masks[y1:y2, x1:x2]
		return crop_mask
