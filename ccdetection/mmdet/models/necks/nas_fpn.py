#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch

from ..registry import NECKS
from ..utils import ConvModule


class rcb(ConvModule):
    #Relu-Conv-BN
    def __init__(self, in_channels, out_channels, kernel_size,
				 stride=1, padding=0, dilation=1, groups=1, bias='auto',
				 conv_cfg=None, norm_cfg=None, activation='relu',
				 inplace=True):
        super(rcb, self).__init__(in_channels,out_channels,kernel_size,stride,
                                  padding,dilation,groups,bias,conv_cfg,norm_cfg,activation='relu',inplace=True)

    def forward(self,x):
        x = self.conv(self.activate(x))
        x = self.norm(x)
        return x

#------------------------------------------------------------------------------
#  FPN
#------------------------------------------------------------------------------
@NECKS.register_module
class NASFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs,
                 start_level=0, end_level=-1,
                 add_extra_convs=False, #Usage: If true, use conv(stride=2). If False, use MaxPool
                 activation=None,
                 conv_cfg=None, norm_cfg=dict(type='BN')):

        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=True)
            self.lateral_convs.append(l_conv)

        if self.add_extra_convs:
            self.conv_P6 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=True)
            self.conv_P7 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=True)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # NAS rcb modules:
        self.rcb_1 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.rcb_2 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.rcb_3 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.rcb_4 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.rcb_5 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.rcb_6 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)
        self.rcb_7 = rcb(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        outs = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        P3, P4, P5 = outs[0], outs[1], outs[2]

        # part 2: add extra levels P6 and P7
        if self.add_extra_convs:
            P6 = self.conv_P6(P5)
            P7 = self.conv_P7(P6)
        else:
            P6 = self.maxpool(P5)
            P7 = self.maxpool(P6)

        #Build merging net in NAS-FPN
        GP_P6_P4 = self.gp(P6, P4)
        GP_P6_P4_RCB = self.rcb_1(GP_P6_P4)
        SUM1 = self.sum_fm(GP_P6_P4_RCB, P4)
        SUM1_RCB = self.rcb_2(SUM1)
        SUM2 = self.sum_fm(SUM1_RCB, P3)
        SUM2_RCB = self.rcb_3(SUM2)  # P3 
        SUM3 = self.sum_fm(SUM2_RCB, SUM1_RCB)
        SUM3_RCB = self.rcb_4(SUM3)  # P4 
        SUM3_RCB_GP = self.gp(SUM2_RCB, SUM3_RCB)
        SUM4 = self.sum_fm(SUM3_RCB_GP, P5)
        SUM4_RCB = self.rcb_5(SUM4)  # P5 
        SUM4_RCB_GP = self.gp(SUM1_RCB, SUM4_RCB)
        SUM5 = self.sum_fm(SUM4_RCB_GP, P7)
        SUM5_RCB = self.rcb_6(SUM5)  # P7 

        h, w = P6.shape[2], P6.shape[3]
        htm, wtm = SUM5_RCB.shape[2], SUM5_RCB.shape[3]
        scale_factor = (float(h/htm), float(w/wtm))

        SUM5_RCB_resize = F.interpolate(SUM5_RCB, scale_factor=scale_factor, mode="bilinear", align_corners=True)
        SUM4_RCB_GP1 = self.gp(SUM4_RCB, SUM5_RCB_resize)
        SUM4_RCB_GP1_RCB = self.rcb_7(SUM4_RCB_GP1)  # P6

        pyramid_list = [SUM2_RCB, SUM3_RCB, SUM4_RCB, SUM4_RCB_GP1_RCB, SUM5_RCB]

        return tuple(pyramid_list)

    def gp(self, fm1, fm2):
        """
        Arguments:
            fm1: higher level feature layer
            fm2: lower level feature layer
        Returns:
            output: feature layer after global pooling operation
        """
        h2, w2 = fm2.shape[2], fm2.shape[3]
        global_ctx = torch.mean(fm1,dim = (2,3), keepdim = True)
        global_ctx = global_ctx.sigmoid()

        h1, w1 = fm1.shape[2], fm1.shape[3]
        scale_factor = (float(h2/h1), float(w2/w1))
        output = (global_ctx * fm2) + F.interpolate(fm1, scale_factor=scale_factor, mode="bilinear", align_corners=True)
        return output

    def sum_fm(self, fm1, fm2):
        h2, w2 = fm2.shape[2], fm2.shape[3]
        h1, w1 = fm1.shape[2], fm1.shape[3]
        scale_factor = (float(h2/h1), float(w2/w1))
        output = fm2 + F.interpolate(fm1, scale_factor=scale_factor, mode="bilinear", align_corners=True)
        return output

