import torch
import torch.nn.functional as F

from mmdet.core import bbox2delta, bbox_overlaps, delta2bbox, force_fp32
from ..registry import HEADS
from .retina_head import RetinaHead


@HEADS.register_module
class FreeAnchorRetinaHead(RetinaHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 iou_threshold=0.3,
                 pre_anchor_topk=50,
                 bbox_threshold=0.6,
                 **kwargs):
        super(FreeAnchorRetinaHead,
              self).__init__(num_classes, in_channels, stacked_convs,
                             octave_base_scale, scales_per_octave, conv_cfg,
                             norm_cfg, **kwargs)

        self.iou_threshold = iou_threshold
        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_threshold = bbox_threshold
        self.focal_loss_alpha = self.loss_cls.alpha
        self.focal_loss_gamma = self.loss_cls.gamma

        self.positive_bag_loss_func = positive_bag_loss
        self.negative_bag_loss_func = focal_loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, _ = self.get_anchors(featmap_sizes, img_metas)
        anchors = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls.permute(0, 2, 3,
                        1).reshape(cls.size(0), -1, self.cls_out_channels)
            for cls in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4)
            for bbox_pred in bbox_preds
        ]
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)

        cls_prob = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        positive_losses = []
        for _, (anchors_, gt_labels_, gt_bboxes_, cls_prob_,
                bbox_preds_) in enumerate(
                    zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds)):
            gt_labels_ -= 1

            with torch.set_grad_enabled(False):
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                pred_boxes = delta2bbox(anchors_, bbox_preds_,
                                        self.target_means, self.target_stds)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = bbox_overlaps(gt_bboxes_, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = self.bbox_threshold
                t2 = object_box_iou.max(
                    dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(
                    min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels_.size(0)
                indices = torch.stack(
                    [torch.arange(num_obj).type_as(gt_labels_), gt_labels_],
                    dim=0)
                object_cls_box_prob = torch.sparse_coo_tensor(
                    indices, object_box_prob)

                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                """
                from "start" to "end" implement:
                image_box_iou = torch.sparse.max(object_cls_box_prob,
                                                 dim=0).t()

                """
                # start
                box_cls_prob = torch.sparse.sum(
                    object_cls_box_prob, dim=0).to_dense()

                indices = torch.nonzero(box_cls_prob).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(
                        anchors_.size(0),
                        self.cls_out_channels).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (gt_labels_.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor(
                            [0]).type_as(object_box_prob)).max(dim=0).values

                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]),
                        nonzero_box_prob,
                        size=(anchors_.size(0),
                              self.cls_out_channels)).to_dense()
                # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = bbox_overlaps(gt_bboxes_, anchors_)
            _, matched = torch.topk(
                match_quality_matrix,
                self.pre_anchor_topk,
                dim=1,
                sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2,
                gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk,
                                                 1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors_[matched]
            matched_object_targets = bbox2delta(
                matched_anchors,
                gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors),
                self.target_means, self.target_stds)
            loss_bbox = self.loss_bbox(
                bbox_preds_[matched],
                matched_object_targets,
                reduction_override='none').sum(-1)
            matched_box_prob = torch.exp(-loss_bbox)

            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes_)
            positive_losses.append(
                self.positive_bag_loss_func(
                    matched_cls_prob * matched_box_prob, dim=1))
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss_func(
            cls_prob * (1 - box_prob), self.focal_loss_gamma) / max(
                1, num_pos * self.pre_anchor_topk)

        losses = {
            "loss_retina_positive": positive_loss * self.focal_loss_alpha,
            "loss_retina_negative":
            negative_loss * (1 - self.focal_loss_alpha),
        }
        return losses


def positive_bag_loss(logits, *args, **kwargs):
    # bag_prob = Mean-max(logits)
    weight = 1 / torch.clamp(1 - logits, 1e-12, None)
    weight /= weight.sum(*args, **kwargs).unsqueeze(dim=-1)
    bag_prob = (weight * logits).sum(*args, **kwargs)
    # positive_bag_loss = -log(bag_prob)
    return F.binary_cross_entropy(
        bag_prob, torch.ones_like(bag_prob), reduction='none')


def focal_loss(logits, gamma):
    return torch.sum(logits**gamma * F.binary_cross_entropy(
        logits, torch.zeros_like(logits), reduction='none'))
