import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def mask_focal_loss(pred, target, label, alpha=0.25, gamma=2.0, reduction='mean', avg_factor=None):
	assert reduction == 'mean' and avg_factor is None
	num_rois = pred.size()[0]
	inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
	pred = pred[inds, label].squeeze(1)

	loss = sigmoid_focal_loss(
		pred, target, weight=None,
		alpha=alpha, gamma=gamma,
		reduction=reduction,
		avg_factor=avg_factor,
	)
	return loss


@LOSSES.register_module
class SegmFocalLoss(nn.Module):

	def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
		super(SegmFocalLoss, self).__init__()
		assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
		self.use_sigmoid = use_sigmoid
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction
		self.loss_weight = loss_weight

	def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None):
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (reduction_override if reduction_override else self.reduction)

		loss_cls = self.loss_weight * mask_focal_loss(
			cls_score, label, weight,
			alpha=self.alpha, gamma=self.gamma,
			reduction=reduction, avg_factor=avg_factor,
		)
		return loss_cls
