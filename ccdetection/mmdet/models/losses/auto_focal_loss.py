#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from scipy.stats import norm

from ..registry import LOSSES
from .utils import weighted_loss
from .focal_loss import sigmoid_focal_loss


#------------------------------------------------------------------------------
#  _expand_binary_labels
#------------------------------------------------------------------------------
def _expand_binary_labels(labels, label_channels):
	bin_labels = labels.new_full((labels.size(0), label_channels), 0)
	inds = torch.nonzero(labels >= 1).squeeze()
	if inds.numel() > 0:
		bin_labels[inds, labels[inds] - 1] = 1
	return bin_labels


#------------------------------------------------------------------------------
#  AutoFocalLoss
#------------------------------------------------------------------------------
@LOSSES.register_module
class AutoFocalLoss(nn.Module):

	def __init__(self, use_sigmoid=False, loss_weight=1.0, gamma=2.0, alpha=0.5, reduction='mean'):
		super(AutoFocalLoss, self).__init__()
		assert use_sigmoid is True, 'Only sigmoid focaloss supported now.'
		self.use_sigmoid = use_sigmoid
		self.loss_weight = loss_weight
		self.gamma = gamma
		self.alpha = alpha
		self.cls_criterion = sigmoid_focal_loss
		self.p_avg = 1e-3
		self.reduction = reduction

	def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (reduction_override if reduction_override else self.reduction)

		if self.use_sigmoid:
			cls_channels = pred.shape[1]
			bin_target = _expand_binary_labels(target, cls_channels)
			bin_target = bin_target.float()

			p = pred.sigmoid()
			p_score = p * bin_target
			p_sum = torch.sum(p_score,1)
			n_non0 = torch.nonzero(p_sum).shape[0]

			if n_non0 !=0:
				p_avg = (torch.sum(p_sum)/n_non0).float().item()
			else:
				p_avg = self.p_avg

			if self.p_avg ==1e-3:
				self.p_avg = p_avg
			else:
				self.p_avg = 0.05*p_avg + 0.95*self.p_avg

			self.gamma = -math.log(self.p_avg)

			loss_cls = 2.0 * self.loss_weight * sigmoid_focal_loss(
				pred, target, weight,
				gamma=self.gamma, alpha=self.alpha,
				reduction=reduction, avg_factor=avg_factor)
		else:
			raise NotImplementedError

		return loss_cls


#------------------------------------------------------------------------------
#  AutoFocalLoss_Regression
#------------------------------------------------------------------------------
@LOSSES.register_module
class AutoFocalLoss_Regression(nn.Module):
	def __init__(self, loss_weight=1.0):
		super(AutoFocalLoss_Regression, self).__init__()
		self.loss_weight = loss_weight
		self.p_correct = None

	@weighted_loss
	def focaloss_regression(self, pred, target):
		d = torch.abs(pred-target)
		sigma = d.std()
		var = sigma ** 2
		dvar = (d/var).detach().cpu()

		p_correct_ = 1 - (norm.cdf(dvar)-norm.cdf(-dvar))
		p_correct = torch.tensor(p_correct_.mean()).cuda()

		if self.p_correct is None:
			self.p_correct = p_correct
		else:
			self.p_correct = 0.05*p_correct+0.95*self.p_correct

		gamma = -torch.log(self.p_correct).float().item()
		loss = d * (1-p_correct)**gamma + torch.log(var+1)
		return loss

	def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (reduction_override if reduction_override else self.reduction)
		loss_bbox = self.loss_weight * self.focaloss_regression(
											pred, target, weight,
											reduction=reduction, avg_factor=avg_factor, **kwargs)
		return loss_bbox
