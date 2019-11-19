import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
from timm.models import *


@BACKBONES.register_module
class EfficientNet(nn.Module):
    def __init__(self, model_name='b0'):
        # if model_name == 'b0':
        self.out_idx=[1,2,4,6]

        super(EfficientNet, self).__init__()
        _model_cls = eval(f'efficientnet_{model_name}')
        self.eff_net = _model_cls()

        x = torch.randn(2,3,256,256)
        outs = self.forward(x)
        for i, _ in enumerate(outs): print(i, _.shape)

        


    def init_weights(self, pretrained=None):
        pass
            
    def forward(self, x):
        outs =  self.eff_net(x)
        # import ipdb; ipdb.set_trace()
        return [outs[i] for i in self.out_idx]
