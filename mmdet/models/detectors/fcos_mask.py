import torch
from ..registry import DETECTORS
from .fcos import FCOS
from .mask_single_stage import MaskSingleStateDetector
from .. import builder


@DETECTORS.register_module
class FCOSMask(MaskSingleStateDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(FCOSMask, self).__init__(
            backbone,
            neck,
            bbox_head,
            mask_roi_extractor,
            mask_head,
            train_cfg,
            test_cfg,
            pretrained,
        )

        self.mask_roi_extractor = builder.build_roi_extractor(
            mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)

    # def __init__(self,
    #              backbone,
    #              neck,
    #              bbox_head,
    #              train_cfg=None,
    #              test_cfg=None,
    #              pretrained=None):
    #     super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
    #                                test_cfg, pretrained)
