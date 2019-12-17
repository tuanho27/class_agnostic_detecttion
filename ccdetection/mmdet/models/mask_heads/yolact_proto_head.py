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
				loss_combine_protonet=dict(type='CrossEntropyLoss', use_sigmoid=True,loss_weight=1.0),
				loss_rmi = dict(type='RMILoss', num_classes=81, loss_weight_lambda=0.5, lambda_way=1),
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
					padding=padding,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg)
		# self.conv1x1_proto = nn.Conv2d(32*2, 32, 3, padding=1)
		self.last_conv = nn.Conv2d(self.conv_out_channels, num_classes, kernel_size=1, stride=1)

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
		# normal_init(self.conv1x1_proto, std=0.01)

	@auto_fp16()
	# def forward(self, feats, protonet_coff, img):
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

		out_proto = torch.tanh(self.conv1x1(x))
		# for protonet cofficient
		# protonet_coff = self.conv1x1_proto(protonet_coff)
		return out_proto #,  protonet_coff

		# for rmi loss
		# logit = self.last_conv(out_proto)
		# logit_4D = F.interpolate(logit, size=img.size()[2:], mode='bilinear', align_corners=True)
		# return out_proto,  protonet_coff, logit_4D


	@force_fp32(apply_to=('mask_preds','mask_coffs','mask_targets'))
	def loss(self, mask_targets, mask_preds, mask_coffs, cls_scores, bbox_preds, extra_data):
	# def loss(self, mask_targets, mask_preds, mask_coff, logit_4D, mask_target_4D):
		# out_coff_resize = F.interpolate(mask_coff,
							# size=(mask_preds.shape[2],mask_preds.shape[3]),
							# mode=self.upsample_method)
		# out = torch.matmul(mask_preds,out_coff_resize)
		# out_new = torch.mean(torch.tanh(out), dim=1, keepdim=True)
		# out_new = F.interpolate(out_new,

		# out_new = torch.mean(mask_preds, dim=1, keepdim=True)
		# out_new = F.interpolate(out_new,
							# size=(mask_targets.shape[2],mask_targets.shape[3]),
							# mode=self.upsample_method)

		
		# Flatten tensor
		featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
		# print("Feature Map size: ",featmap_sizes)
		all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,bbox_preds[0].device)
		labels, _, _ = self.polar_target(all_level_points, extra_data)
		num_imgs = cls_scores[0].size(0)
		
		flatten_mask_coffs = [
            mask_coff.permute(0, 2, 3, 1).reshape(-1, 32)
            for mask_coff in mask_coffs
        ]
		flatten_mask_coffs = torch.cat(flatten_mask_coffs) 
		flatten_labels = torch.cat(labels).long()
		pos_inds = flatten_labels.nonzero().reshape(-1)
		pos_mask_coffs = flatten_mask_coffs[pos_inds]

		mask_preds_assembly = []
		for mask in mask_preds:
			out = torch.sigmoid(torch.matmul(mask.permute(1,2,0), pos_mask_coffs.t()))
			mask_preds_assembly.append(out)

		mask_preds_assembly = torch.mean(torch.stack(mask_preds_assembly), dim=3, keepdim=True)
		mask_preds_assembly = F.interpolate(mask_preds_assembly.permute(0,3,1,2),
					size=(mask_targets.shape[2],mask_targets.shape[3]),
					mode=self.upsample_method)

		# mask_preds = mask_preds.permute(0, 2, 3, 1).reshape(-1, 1)
		mask_preds_assembly = mask_preds_assembly.permute(0, 2, 3, 1).view(-1,1)
		mask_targets = mask_targets.permute(0, 2, 3, 1).reshape(-1, 1)
		# Compute loss
		loss_combine_protonet = self.loss_combine_protonet(mask_preds_assembly, mask_targets)
		return {'loss_combine_protonet':loss_combine_protonet}

		# loss_rmi = self.loss_rmi(logit_4D, mask_target_4D).mean()
		# return {'loss_combine_protonet':loss_combine_protonet}, {'loss_rmi':loss_rmi}


	def get_points(self, featmap_sizes, dtype, device):
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