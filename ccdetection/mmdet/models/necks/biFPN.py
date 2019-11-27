import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import copy

class WeightSum(nn.Module):
    def __init__(self,length):
        super(WeightSum,self).__init__()
        assert length==2 or length==3
        self.l = length
        self.w = nn.Parameter(torch.ones(length,requires_grad=True))

    def forward(self,x):
        assert isinstance(x,list) and len(x)==self.l
        w= self.w/(self.w.sum()+1e-6)
        if self.l==2:
            return w[0]*x[0]+w[1]*x[1]
        else:
            return w[0]*x[0]+w[1]*x[1]+w[2]*x[2]

class BiFPN(nn.Module):
    def __init__(self,
                 out_channels,
                 num_outs=5,
                 fpn_conv_groups=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):

        super(BiFPN, self).__init__()
        self.num_outs = num_outs
        # Build Top-down, Bottom-Up path
        self.fpn_down = nn.ModuleList()
        self.fpn_up = nn.ModuleList()
        for i in range(num_outs-1):
            fpn_down = ConvModule(
                out_channels,
                out_channels,
                3,
                groups=fpn_conv_groups,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)
            fpn_up = copy.deepcopy(fpn_down)

            self.fpn_down.append(fpn_down)
            self.fpn_up.append(fpn_up)

        # Fuse layers
        self.fuse_td = nn.ModuleList()
        self.fuse_out = nn.ModuleList()
        for i in range(num_outs-1):
            self.fuse_td.append(WeightSum(2))
            if i == num_outs-2:
                self.fuse_out.append(WeightSum(2))
            else:
                self.fuse_out.append(WeightSum(3))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    # @auto_fp16()
    def forward(self, P_in):
        assert len(P_in) == self.num_outs
        # build from top-down path
        P_td = [P.clone() for P in P_in]
        for i in range(self.num_outs-1, 1, -1):
            P_up = F.interpolate(P_td[i], scale_factor=2, mode='nearest')
            P_fuse = self.fuse_td[i-1]([P_td[i - 1],P_up])
            P_td[i - 1] = self.fpn_down[i-1](P_fuse)

        # build from bottom-up path
        P_out = P_td # Just change name
        for i in range(1,self.num_outs):
            P_down = F.avg_pool2d(P_out[i-1],kernel_size=2,stride=2)
            if i==self.num_outs-1:
                P_fuse = self.fuse_out[i-1]([P_in[i],P_down])
            else:
                P_fuse = self.fuse_out[i-1]([P_in[i],P_out[i],P_down])
            P_out[i] = self.fpn_up[i-1](P_fuse)

        return P_out

@NECKS.register_module
class StackBiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 fpn_stack=2,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 fpn_conv_groups=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(StackBiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fpn_conv_groups = fpn_conv_groups if fpn_conv_groups !=-1 else out_channels
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_levfpn_conv_groupsel = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        # Build P3-P5 from Input
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # Build P6-P7 layers (e.g., RetinaNet)
        self.extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.extra_levels >= 1:
            self.extra_convs = nn.ModuleList()
            for i in range(self.extra_levels):
                if self.add_extra_convs:
                    if i == 0 and self.extra_convs_on_inputs:
                        in_channels = self.in_channels[self.backbone_end_level - 1]
                    else:
                        in_channels = out_channels
                    extra_fpn_conv = ConvModule(
                        in_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False)
                    self.extra_convs.append(extra_fpn_conv)
                else:
                    self.extra_convs.append(nn.MaxPool2d(kernel_Size=2,stride=2))

        self.stack_bifpn = nn.ModuleList([BiFPN(out_channels,
                                num_outs,
                                self.fpn_conv_groups,
                                conv_cfg,
                                norm_cfg,
                                activation) ]*fpn_stack)
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        for m in self.stack_bifpn:
            m.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals P3,P4,P5
        P_out = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build extra-conv P6,P7
        if self.extra_levels >= 1:
            if self.extra_convs_on_inputs and self.add_extra_convs:
                in_feature = inputs[-1]
            else:
                in_feature = P_out[-1]
            for extra_conv in self.extra_convs:
                in_feature = extra_conv(in_feature)
                P_out.append(in_feature)
        

        for m in self.stack_bifpn:
            P_out = m(P_out)

        return tuple(P_out)
