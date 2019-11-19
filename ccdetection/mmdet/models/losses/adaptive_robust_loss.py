#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn

import numpy as np

from ..registry import LOSSES
from .utils import weighted_loss
from .adaptive import AdaptiveLossFunction


#------------------------------------------------------------------------------
#  _expand_binary_labels
#------------------------------------------------------------------------------
def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


#------------------------------------------------------------------------------
#  adapt_loss
#------------------------------------------------------------------------------
@weighted_loss
def adapt_loss(pred, target):
    loss = pred
    return loss


#------------------------------------------------------------------------------
#  AdaptiveRobustLoss_C
#------------------------------------------------------------------------------
@LOSSES.register_module
class AdaptiveRobustLoss_C(AdaptiveLossFunction):
    def __init__(self, num_classes, use_sigmoid=True, loss_weight=1.0, reduction='mean',
                float_dtype=np.float32, device='cuda', alpha_lo=0.001, alpha_hi=1.999,
                alpha_init=None, scale_lo=1e-5, scale_init=1.0):

        num_dims = num_classes - 1
        super(AdaptiveRobustLoss_C, self).__init__(
            num_dims, float_dtype, device,
            alpha_lo, alpha_hi, alpha_init, scale_lo, scale_init)
        assert use_sigmoid is True, 'Only sigmoid focaloss supported now.'
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if self.use_sigmoid:
            cls_channels = pred.shape[1]
            if pred.dim() != target.dim():
                bin_target, weight = _expand_binary_labels(target, weight, cls_channels)
            bin_target, weight = bin_target.float(), weight.float()

            p = pred.sigmoid()
            p_score = bin_target*p
            p_sum = torch.sum(p_score,1)
            n_non0 = torch.nonzero(p_sum).shape[0]

            if n_non0 !=0:
                p_avg = (torch.sum(p_sum)/n_non0).float().item()
            else:
                p_avg = self.p_avg

            if self.p_avg is None:
                self.p_avg = p_avg
            else:
                self.p_avg = 0.05*p_avg+0.95*self.p_avg
            
            loss = self.lossfun(self.p_avg)
            loss_cls = adapt_loss(loss, target, weight, reduction=reduction, avg_factor=avg_factor)

        return self.loss_weight * loss_cls


#------------------------------------------------------------------------------
#  AdaptiveRobustLoss_R
#------------------------------------------------------------------------------
@LOSSES.register_module
class AdaptiveRobustLoss_R(AdaptiveLossFunction):
    def __init__(self, num_dims, loss_weight=1.0, reduction='mean',
                float_dtype=np.float32, device='cuda', alpha_lo=0.001, alpha_hi=1.999,
                alpha_init=None, scale_lo=1e-5, scale_init=1.0):

        super(AdaptiveRobustLoss_R, self).__init__(
            num_dims, float_dtype, device, alpha_lo, alpha_hi,
            alpha_init, scale_lo, scale_init)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        x = torch.abs(pred - target)
        print("x.dtype", x.dtype)
        loss = self.lossfun(x)

        loss_bbox = adapt_loss(loss, target, weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return self.loss_weight * loss_bbox
