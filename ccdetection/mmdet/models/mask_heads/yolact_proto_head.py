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
class YolactProtoHead(nn.Module):
	def __init__(self,
				num_convs=3,
				num_convs_post=1,
				in_channels=256,
				conv_kernel_size=3,
				conv_out_channels=32,
				input_index=0,
				upsample_method='bilinear',
				upsample_ratio=2,
				num_classes=1,
				conv_cfg=None,
				norm_cfg=None,
				loss_combine_protonet=dict(type='CrossEntropyLoss', loss_weight=1.0),
				strides=(4, 8, 16, 32, 64),
				regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)),

				):
			
		super(YolactProtoHead, self).__init__()	
		self.num_convs = num_convs
		self.num_convs_post = num_convs_post
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
		self.convs = nn.ModuleList()
		self.loss_combine_protonet = build_loss(loss_combine_protonet)

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
					padding=padding,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg)
		self.conv1x1_proto = nn.Conv2d(32*5, 32, 3, padding=1)

	def init_weights(self):
		for m in [self.upsample, self.convs, self.conv1x1, self.convs_post]:
			if m is None:
				continue
			if hasattr(m, 'weight'):
				nn.init.kaiming_normal_(
					m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)

				nn.init.constant_(m.bias, 0)
		## add for protonet
		normal_init(self.conv1x1_proto, std=0.01)

	@auto_fp16()
	def forward(self, feats, protonet_coff):
		x = feats[self.input_index]

		for conv in self.convs:
			x = conv(x)
		if self.upsample_method == 'deconv':
			x = self.upsample(x)
			x = self.relu(x)
		else:
			x = F.interpolate(x, 
							size=self.in_channels,
							mode=self.upsample_method,
							align_corners=True)
		
		for conv in self.convs_post:
			x = conv(x)

		out_proto = self.conv1x1(x)
		protonet_coff = [self.conv1x1_proto(protonet_coff)]

		return out_proto, protonet_coff


	@force_fp32(apply_to=('out_mask','outs_coff','mask_targets'))
	def loss(self, mask_targets, out_mask, outs_coff, extra_data):
	
		out_coff = outs_coff[0] 		
		out_coff_resize = F.interpolate(out_coff,
							size=(out_mask.shape[2],out_mask.shape[3]),
							mode=self.upsample_method)
		out = torch.matmul(out_mask,out_coff_resize)
		out_new = torch.mean(torch.tanh(out), dim=1, keepdim=True)
		out_new = F.interpolate(out_new,
							size=(mask_targets.shape[2],mask_targets.shape[3]),
							mode=self.upsample_method)

		# Flatten tensor
		mask_pred = out_new.permute(0, 2, 3, 1).reshape(-1, 1)
		mask_targets = mask_targets.permute(0, 2, 3, 1).reshape(-1, 1)

		# Compute loss
		loss_combine_protonet = self.loss_combine_protonet(mask_pred, mask_targets)

		return {'loss_combine_protonet':loss_combine_protonet}
