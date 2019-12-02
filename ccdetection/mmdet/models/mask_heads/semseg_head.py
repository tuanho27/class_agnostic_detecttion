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

	def init_weights(self):
		for m in [self.upsample, self.conv_logits]:
			if m is None:
				continue
			if hasattr(m, 'weight'):
				nn.init.kaiming_normal_(
					m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

	@auto_fp16()
	def forward(self, feats):
		x = feats[self.input_index]
		for conv in self.convs:
			x = conv(x)
		if self.upsample is not None:
			x = self.upsample(x)
			if self.upsample_method == 'deconv':
				x = self.relu(x)
		mask_pred = self.conv_logits(x)
		return mask_pred

	@force_fp32(apply_to=('mask_pred', ))
	def loss(self, mask_pred, mask_targets):
		# Flatten tensor
		num_classes = mask_pred.shape[1]
		mask_pred = mask_pred.permute(0, 2, 3, 1).reshape(-1, num_classes)
		mask_targets = mask_targets.permute(0, 2, 3, 1).reshape(-1, num_classes)

		# Compute loss
		loss_semseg = self.loss_mask(mask_pred, mask_targets)
		return {'loss_semseg': loss_semseg}

	def get_seg_masks(self, mask_pred, ori_shape, scale_factor, rescale, threshold=0.5):
		if rescale:
			mask_pred = F.interpolate(mask_pred, size=ori_shape[:2], mode='bilinear', align_corners=True)

		mask_pred = mask_pred.sigmoid().cpu().numpy()
		mask_pred = (mask_pred > threshold).astype('uint8')
		return mask_pred
