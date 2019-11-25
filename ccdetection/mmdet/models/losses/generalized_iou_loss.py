#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch.nn as nn
import torch 

from ..registry import LOSSES
from .utils import weighted_loss

def bbox_overlaps_giou(bboxes1, bboxes2, is_aligned=False):
    """Calculate generalized IOU between two set of bboxes.
    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        gen_ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = torch.abs(bboxes1[:, 2] - bboxes1[:, 0] + 1) * torch.abs(bboxes1[:, 3] - bboxes1[:, 1] + 1)
        area2 = torch.abs(bboxes2[:, 2] - bboxes2[:, 0] + 1) * torch.abs(bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1 + area2 - overlap)       
        # Find convex enclosed rectangle of two bounding boxes
        ltc = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rbc = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
        whc = torch.abs(rbc - ltc + 1)  # [rows, 2]
        area_c = whc[:, 0] * whc[:, 1]

        gen_ious = ious - (area_c - (area1 + area2 - overlap)) / area_c
        if ((area_c - (area1 + area2 - overlap))<0.0).any():
            gen_ious = ious
            # print('not good')
            # print('area_c: ', area_c[torch.nonzero((area_c - (area1 + area2 - overlap))<0.0)])
            # print('U: ', (area1 + area2 - overlap)[torch.nonzero((area_c - (area1 + area2 - overlap))<0.0)])          
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1[:, None] + area2 - overlap)

        # Find convex enclosed rectangle of two bounding boxes
        ltc = torch.min(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, 2]
        rbc = torch.max(bboxes1[:, None, 2:], bboxes2[:, 2:])

        whc = (rbc - ltc + 1).clamp(min=0)  # [rows, 2]
        area_c = whc[:, 0] * whc[:, 1]
        gen_ious = ious - (area_c - (area1 + area2 - overlap)) / area_c

    return gen_ious

#------------------------------------------------------------------------------
#  generalized_iou_loss
#------------------------------------------------------------------------------
@weighted_loss
def generalized_iou_loss(pred, target, eps=1e-6):   
    gen_ious = bbox_overlaps_giou(pred, target, is_aligned=True)
    loss = 1-gen_ious
    return loss


#------------------------------------------------------------------------------
#  GeneralizedIOULoss
#------------------------------------------------------------------------------
@LOSSES.register_module
class GeneralizedIOULoss(nn.Module):
    def __init__(self, eps = 1e-6, loss_weight=1.0):
        super(GeneralizedIOULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * generalized_iou_loss(
            pred, target, weight, eps=self.eps,
            reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss
