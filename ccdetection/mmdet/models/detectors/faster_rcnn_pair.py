from ..registry import DETECTORS
from .two_stage_pair import TwoStagePairDetector


@DETECTORS.register_module
class FasterRCNNPair(TwoStagePairDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 siamese_matching_head,
                 relation_matching_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNNPair, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            siamese_matching_head=siamese_matching_head,
            relation_matching_head=relation_matching_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
