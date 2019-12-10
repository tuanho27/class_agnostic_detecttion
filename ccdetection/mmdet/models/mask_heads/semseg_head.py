import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule

INF = 1e8

@HEADS.register_module
class SemSegHead(nn.Module):

	def __init__(self,
				num_convs=4,
				in_channels=256,
				conv_kernel_size=3,
				conv_out_channels=256,
				input_index=0,
				upsample_method='bilinear',
				upsample_ratio=2,
				num_classes=1,
				conv_cfg=None,
				norm_cfg=None,
				loss_mask=dict(type='CrossEntropyLoss', loss_weight=1.0),
				loss_cls_combine=dict(
					 type='FocalLoss',
					 use_sigmoid=True,
					 gamma=2.0,
					 alpha=0.25,
					 loss_weight=1.0),
				strides=(4, 8, 16, 32, 64),
				regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)),
		):

		super(SemSegHead, self).__init__()
		if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
			raise ValueError('Invalid upsample method {}, accepted methods are "deconv", "nearest", "bilinear"'.format(upsample_method))

		self.num_convs = num_convs
		self.in_channels = in_channels
		self.conv_kernel_size = conv_kernel_size
		self.conv_out_channels = conv_out_channels
		self.input_index = input_index
		self.upsample_method = upsample_method
		self.upsample_ratio = upsample_ratio
		self.num_classes = num_classes
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.fp16_enabled = False
		self.loss_mask = build_loss(loss_mask)
		self.loss_cls_combine = build_loss(loss_cls_combine)
		self.convs = nn.ModuleList()
		for i in range(self.num_convs):
			in_channels = (
				self.in_channels if i == 0 else self.conv_out_channels)
			padding = (self.conv_kernel_size - 1) // 2
			self.convs.append(
				ConvModule(
					in_channels,
					self.conv_out_channels,
					self.conv_kernel_size,
					padding=padding,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg))
		upsample_in_channels = (
			self.conv_out_channels if self.num_convs > 0 else in_channels)
		if self.upsample_method is None:
			self.upsample = None
		elif self.upsample_method == 'deconv':
			self.upsample = nn.ConvTranspose2d(
				upsample_in_channels,
				self.conv_out_channels,
				self.upsample_ratio,
				stride=self.upsample_ratio)
		else:
			self.upsample = nn.Upsample(
				scale_factor=self.upsample_ratio, mode=self.upsample_method)

		out_channels = self.num_classes
		logits_in_channel = (
			self.conv_out_channels
			if self.upsample_method == 'deconv' else upsample_in_channels)
		self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
		self.relu = nn.ReLU(inplace=True)
		self._convs = ConvModule(
							81,
							80,
							kernel_size=3, 
							stride=1,
							padding=1)
		self.strides = strides
		self.regress_ranges = regress_ranges

	def init_weights(self):
		for m in [self.upsample, self.conv_logits]:
			if m is None:
				continue
			if hasattr(m, 'weight'):
				nn.init.kaiming_normal_(
					m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	@auto_fp16()
	def forward(self, feats, outs):
		x = feats[self.input_index]
		for conv in self.convs:
			x = conv(x)
		if self.upsample is not None:
			x = self.upsample(x)
			if self.upsample_method == 'deconv':
				x = self.relu(x)
		mask_pred = self.conv_logits(x)

		new_outs = []
		for i,out in enumerate(outs[0]):
			# _pool = nn.AdaptiveAvgPool2d((out.shape[2],out.shape[3]))
			# _convs = ConvModule(
			# 			out.shape[1]+1,
			# 			out.shape[1],
			# 			kernel_size=4, 
			# 			stride=2,
			# 			padding=1)
			## try interpolate
			out_cls_scale = torch.cat((out, F.interpolate(mask_pred,size=(out.shape[2],out.shape[3]), mode='nearest')), dim=1)
			# out_cls_scale = torch.cat((out, _pool(mask_pred)), dim=1)
			# out_cls = _convs(out_cls_scale.float().cpu())
			out_cls = self._convs(out_cls_scale)
			new_outs.append(out_cls)
		return mask_pred, new_outs

	@force_fp32(apply_to=('mask_pred','new_outs','bbox_preds'))
	def loss(self, mask_pred, mask_targets, new_outs, bbox_preds, extra_data):
		# Flatten tensor
		num_classes = mask_pred.shape[1]
		mask_pred = mask_pred.permute(0, 2, 3, 1).reshape(-1, num_classes)
		mask_targets = mask_targets.permute(0, 2, 3, 1).reshape(-1, num_classes)

		# featmap_sizes = [featmap.size()[-2:] for featmap in new_outs]
		# all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
		# 								   bbox_preds[0].device)
		# labels, _, _ = self.polar_target(all_level_points, extra_data)
		# flatten_cls = [cls_score.permute(0, 2, 3, 1).reshape(-1, 80) for cls_score in new_outs]
		# flatten_cls_scores = torch.cat(flatten_cls)
		# flatten_labels = torch.cat(labels).long() 
		# pos_inds = flatten_labels.nonzero().reshape(-1)
		# num_pos = len(pos_inds)
		# num_imgs = new_outs[0].size(0)
		# import ipdb; ipdb.set_trace()
		# Compute loss
		loss_semseg = self.loss_mask(mask_pred, mask_targets)
		# loss_combine = self.loss_cls_combine(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)

		# return {'loss_semseg': loss_semseg}, {'loss_combine_cls_seg':loss_combine}
		return {'loss_semseg': loss_semseg}

	def get_seg_masks(self, mask_pred, ori_shape, scale_factor, rescale, threshold=0.5):
		if rescale:
			mask_pred = F.interpolate(mask_pred, size=ori_shape[:2], mode='bilinear', align_corners=True)

		mask_pred = mask_pred.sigmoid().cpu().numpy()
		mask_pred = (mask_pred > threshold).astype('uint8')
		return mask_pred

	def polar_target(self, points, extra_data):
		assert len(points) == len(self.regress_ranges)

		num_levels = len(points)

		labels_list, bbox_targets_list, mask_targets_list = extra_data.values()

		# split to per img, per level
		num_points = [center.size(0) for center in points]
		labels_list = [labels.split(num_points, 0) for labels in labels_list]
		bbox_targets_list = [
			bbox_targets.split(num_points, 0)
			for bbox_targets in bbox_targets_list
		]
		mask_targets_list = [
			mask_targets.split(num_points, 0)
			for mask_targets in mask_targets_list
		]

		# concat per level image
		concat_lvl_labels = []
		concat_lvl_bbox_targets = []
		concat_lvl_mask_targets = []
		for i in range(num_levels):
			concat_lvl_labels.append(
				torch.cat([labels[i] for labels in labels_list]))
			concat_lvl_bbox_targets.append(
				torch.cat(
					[bbox_targets[i] for bbox_targets in bbox_targets_list]))
			concat_lvl_mask_targets.append(
				torch.cat(
					[mask_targets[i] for mask_targets in mask_targets_list]))

		return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_mask_targets


	def get_points(self, featmap_sizes, dtype, device):
		"""Get points according to feature map sizes.

		Args:
			featmap_sizes (list[tuple]): Multi-level feature map sizes.
			dtype (torch.dtype): Type of points.
			device (torch.device): Device of points.

		Returns:
			tuple: points of each image.
		"""
		mlvl_points = []
		for i in range(len(featmap_sizes)):
			mlvl_points.append(
				self.get_points_single(featmap_sizes[i], self.strides[i],
									   dtype, device))
		return mlvl_points

	def get_points_single(self, featmap_size, stride, dtype, device):
		h, w = featmap_size
		x_range = torch.arange(
			0, w * stride, stride, dtype=dtype, device=device)
		y_range = torch.arange(
			0, h * stride, stride, dtype=dtype, device=device)
		y, x = torch.meshgrid(y_range, x_range)
		points = torch.stack(
			(x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
		return points
