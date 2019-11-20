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
        self.custom_frozen()

    def custom_frozen(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad=False

        self.mask_head.train()
        for param in self.mask_head.parameters():
            param.requires_grad=True
        