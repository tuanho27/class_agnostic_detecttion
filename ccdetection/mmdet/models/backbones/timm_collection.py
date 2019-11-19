import numpy as np
import timm
import torch.nn.functional as F
from timm.models.gen_efficientnet import GenEfficientNet as BaseEfficientNet
from timm.models.dpn import DPN
from timm.models.gluon_resnet import GluonResNet
from ..registry import BACKBONES
import torch.nn as nn
import torch
from torch.nn.modules.batchnorm import _BatchNorm

gluon_resnet_pyramid = {
'gluon_resnet18_v1b'    :[64,128,256,512],
'gluon_resnet34_v1b'    :[64,128,256,512],
'gluon_resnet50_v1b'    :[256,512,1024,2048],
'gluon_resnet101_v1b'   :[256,512,1024,2048],
'gluon_resnet152_v1b'   :[256,512,1024,2048],
'gluon_resnet50_v1c'    :[256,512,1024,2048],
'gluon_resnet101_v1c'   :[256,512,1024,2048],
'gluon_resnet152_v1c'   :[256,512,1024,2048],
'gluon_resnet50_v1d'    :[256,512,1024,2048],
'gluon_resnet101_v1d'   :[256,512,1024,2048],
'gluon_resnet152_v1d'   :[256,512,1024,2048],
'gluon_resnet50_v1e'    :[256,512,1024,2048],
'gluon_resnet101_v1e'   :[256,512,1024,2048],
'gluon_resnet152_v1e'   :[256,512,1024,2048],
'gluon_resnet50_v1s'    :[256,512,1024,2048],
'gluon_resnet101_v1s'   :[256,512,1024,2048],
'gluon_resnet152_v1s'   :[256,512,1024,2048],
'gluon_resnext50_32x4d' :[256,512,1024,2048],
'gluon_resnext101_32x4d':[256,512,1024,2048],
'gluon_resnext101_64x4d':[256,512,1024,2048],
'gluon_seresnext50_32x4d':[256,512,1024,2048],
'gluon_seresnext101_32x4d':[256,512,1024,2048],
'gluon_seresnext101_64x4d':[256,512,1024,2048],
'gluon_senet154'    :[256,512,1024,2048]
}

dualpathnet = {
'dpn68' :[144, 320, 704, 832],
'dpn68b':[144, 320, 704, 832],
'dpn92' :[336, 704, 1552, 2688],
'dpn98' :[336, 768, 1728, 2688],
'dpn131':[352, 832, 1984, 2688],
'dpn107':[376, 1152, 2432, 2688]
}

densenet_pyramid = {
'densenet201':[256,512,1792,1920],
'densenet121':[256,512,1024,1024],
'densenet161':[384,768,2112,2208],
'densenet169':[256,512,1280,1664]
}

genetic_efficientnet_pyramid = {
'tf_efficientnet_b0':[24,40,112,320],
'tf_efficientnet_b1':[24,40,112,320],
'tf_efficientnet_b2':[24,48,120,352],
'tf_efficientnet_b3':[32,48,136,384],
'tf_efficientnet_b4':[32,56,160,448],
'tf_efficientnet_b5':[40,64,176,512],
'tf_efficientnet_b6':[40,72,200,576],
'tf_efficientnet_b7':[48,80,224,640],

'efficientnet_b0':[24,40,112,320],
'efficientnet_b1':[24,40,112,320],
'efficientnet_b2':[24,48,120,352],
'efficientnet_b3':[32,48,136,384],
'efficientnet_b4':[32,56,160,448],
'efficientnet_b5':[40,64,176,512],
'efficientnet_b6':[40,72,200,576],
'efficientnet_b7':[48,80,224,640],

'tf_mixnet_s':[24,40,120,200],
'tf_mixnet_m':[32,40,120,200],
'tf_mixnet_l':[40,56,160,264],

'mixnet_s':[24,40,120,200],
'mixnet_m':[32,40,120,200],
'mixnet_l':[40,56,160,264],

'mobilenetv1_100':[128,256,512,1024],
'mobilenetv2_100':[24,32,96,320],
'mobilenetv3_050':[16,24,56,480],
'mobilenetv3_075':[24,32,88,720],
'mobilenetv3_100':[24,40,112,960],

'semnasnet_050':[16,24,56,160],
'semnasnet_075':[24,32,88,240],
'semnasnet_100':[24,40,112,320],
'semnasnet_140':[32,56,160,448],

'mnasnet_small':[16,16,32,144],
'mnasnet_a1':[24,40,112,320],
'mnasnet_b1':[24,40,96,320,],
'mnasnet_050':[16,24,48,160],
'mnasnet_075':[24,32,72,240],
'mnasnet_100':[24,40,96,320],
'mnasnet_140':[32,56,136,448],

'chamnetv1_100':[48,64,88,104],
'chamnetv2_100':[32,48,56,112],

'fbnetc_100':[24,32,112,352],
'spnasnet_100':[24,40,96,320],
}

timm_channel_pyramid = {**densenet_pyramid, **genetic_efficientnet_pyramid, **dualpathnet, **gluon_resnet_pyramid}

def get_default_feature_index(scale):
    if isinstance(scale, list):
        scale = np.array(scale)

    idxs = []
    # print(scale)
    for s in [4, 8, 16, 32]:
        if s > max(scale):
            break
        idx = np.where(scale == s)[0].max()
        idxs.append(idx)
    return idxs


@BACKBONES.register_module
class TimmCollection(nn.Module):
    def __init__(self, model_name, pretrained=True, feature_idxs=None, norm_eval=True):
        super(TimmCollection, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if norm_eval:
            for m in self.model.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


        if feature_idxs is None:
            inputs = torch.zeros([2, 3, 256, 256])
            outs = self.model.extract_features(inputs, None)
            scale = [int(256/_.shape[-1]) for _ in outs]
            self.feature_idxs  = get_default_feature_index(scale)
        else:
            self.feature_idxs = feature_idxs

    def forward(self, x):
        outs = self.model.extract_features(x, feature_idxs = self.feature_idxs)
        #return outs
        #import ipdb; ipdb.set_trace()
        return tuple([out for out in outs])


    def init_weights(self, pretrained):
        if pretrained:
            self.load_state_dict(pretrained)
        # if pretrained:
