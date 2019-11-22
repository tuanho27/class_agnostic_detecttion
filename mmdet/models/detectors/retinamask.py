import torch
from ..registry import DETECTORS
from .mask_single_stage import MaskSingleStateDetector

@DETECTORS.register_module
class RetinaMask(MaskSingleStateDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaMask, self).__init__(backbone, neck, bbox_head,mask_roi_extractor,mask_head,train_cfg,
                                        test_cfg, pretrained)
