import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


def mask_focal_loss(pred, target, label, alpha=0.25, gamma=2.0, reduction='mean', avg_factor=None):
	assert reduction == 'mean' and avg_factor is None
	num_rois = pred.size()[0]
	inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
	pred = pred[inds, label].squeeze(1)

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = focal_weight * F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class SegmFocalLoss(nn.Module):

    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

	def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (reduction_override if reduction_override else self.reduction)

		loss_cls = self.loss_weight * mask_focal_loss(
            cls_score, label, weight,
            alpha=self.alpha, gamma=self.gamma,
            reduction=reduction, avg_factor=avg_factor,
            **kwargs,
        )
		return loss_cls
