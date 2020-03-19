import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule
import math
from mmcv.cnn import normal_init
from scipy.spatial import distance as dist
from time import time
from ..losses import accuracy
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class SiameseMatching(nn.Module):
    """
    Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Positive and negative pairs
    """
    def __init__(self, margin=1e-3, 
                       lamb = 2,
                       in_channels=256,
                       feat_channels=128,
                       cls_out_channels = 1, # background and foreground classes
                       num_conv = 3,
                       dist_mode='cosine',
                       loss_siamese=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                       loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                       loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):

        super(SiameseMatching, self).__init__()
        self.margin = margin
        self.lamb = lamb
        self.dist_mode = dist_mode
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_out_channels = cls_out_channels
        self.num_conv = num_conv
        self.loss_siamese = build_loss(loss_siamese)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.siamese_conv = nn.ModuleList()
        for i in range(self.num_conv):
            channel = self.in_channels if i == 0 else self.in_channels*2
            self.siamese_conv.append(
                ConvModule(
                    channel,
                    512,
                    kernel_size=(3,3),
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=None))
        # self.siamese_embedding = nn.Sequential(
        #                          nn.Linear(7*7*in_channels,512),
        #                          nn.ReLU(inplace=True),
        #                          nn.Linear(512,self.feat_channels))
        self.siamese_embedding = nn.Sequential(
                                 nn.Linear(512,256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256,self.feat_channels))
        self.dist = nn.CosineSimilarity(dim=0, eps=1e-7) 

        self.refine_rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.refine_rpn_cls = nn.Conv2d(self.feat_channels,self.cls_out_channels, 1)
        self.refine_rpn_reg = nn.Conv2d(self.feat_channels, self.cls_out_channels*4, 1)
        self.init_weights()

    def init_weights(self):
        for i in range(3):
            if "ReLU" in str(self.siamese_embedding[i]):
                pass 
            else:
                normal_init(self.siamese_embedding[i], std=0.01)

        normal_init(self.refine_rpn_conv, std=0.01)
        normal_init(self.refine_rpn_cls, std=0.01)
        normal_init(self.refine_rpn_reg, std=0.01)

    def forward(self, pairs, pairs_targets, pairs_feats):
        dist_pairs = []
        for i in range(len(pairs)):
            feat = pairs_feats[i]
            for conv in self.siamese_conv:
                feat =conv(feat)
                
            # pairs_fc = [self.siamese_embedding(pairs_feats[i][j].view(-1)) for j in range(2)]
            pairs_fc = [self.siamese_embedding(feat[j].view(-1)) for j in range(2)]
            if self.dist_mode == 'cosine':
                dist_pairs.append(self.dist(pairs_fc[0],pairs_fc[1]))
            else:
                dist_pairs.append((pairs_fc[0] - pairs_fc[1]).pow(2).sum(1))
                # dist_pairs.append(nn.PairwiseDistance(pairs_fc[0],pairs_fc[1]))

        dist_pairs = torch.stack(dist_pairs)
        loss_siamese = self.loss_siamese(dist_pairs[:,None], pairs_targets.long(),avg_factor=len(pairs))
       
        return loss_siamese 

    def forward_test(self, pairs, pairs_feats):
        dist_pairs = []
        for i in range(len(pairs)):
            pairs_fc = [self.siamese_embedding(pairs_feats[i][j].view(-1)) for j in range(2)]
            if self.dist_mode == 'cosine':
                dist_pairs.append(self.dist(pairs_fc[0],pairs_fc[1]))
            else:
                dist_pairs.append((pairs_fc[0] - pairs_fc[1]).pow(2).sum(1))
        dist_pairs = torch.stack(dist_pairs)
        return dist_pairs


@HEADS.register_module
class RelationMatching(nn.Module):
    """
    Relation Matching loss
    Positive and negative pairs
    """
    def __init__(self, in_channels=256,
                       lamb = 2,
                       feat_channels=128,
                       cls_out_channels = 1,
                       num_conv = 4,
                       loss_relation = nn.MSELoss(),
                       loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                       loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):

        super(RelationMatching, self).__init__()
        self.lamb = lamb
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_out_channels = cls_out_channels
        self.num_conv = num_conv
        self.loss_relation = loss_relation
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.relation_module = nn.ModuleList()
        for i in range(self.num_conv):
            channel = self.in_channels * 2 if i == 0 else self.in_channels 
            self.relation_module.append(
                ConvModule(
                    channel,
                    self.in_channels ,
                    kernel_size=(3,3),
                    padding=(2,2),
                    conv_cfg=None,
                    norm_cfg=None))
        self.relation_module_pool = nn.MaxPool2d(kernel_size=(2,2))
        self.relation_module_fc = nn.Sequential(
                                    nn.Linear(self.in_channels , self.feat_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.feat_channels,1))

        self.refine_rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.refine_rpn_cls = nn.Conv2d(self.feat_channels,self.cls_out_channels, 1)
        self.refine_rpn_reg = nn.Conv2d(self.feat_channels, self.cls_out_channels*4, 1)

        self.init_weights()

    def init_weights(self):
        for i in range(3):
            if "ReLU" in str(self.relation_module_fc[i]):
                pass 
            else:
                normal_init(self.relation_module_fc[i], std=0.01)       
        normal_init(self.refine_rpn_conv, std=0.01)
        normal_init(self.refine_rpn_cls, std=0.01)
        normal_init(self.refine_rpn_reg, std=0.01)

    def forward(self, pairs, pairs_targets, pairs_feats):
        N,_,c,h,w = pairs_feats.size()
        feats = pairs_feats.view(N,c*2,h,w)

        for conv in self.relation_module:
            feats =conv(feats)
            feats = self.relation_module_pool(feats)
        feats = F.adaptive_avg_pool2d(feats,1) 
        pairs_fc = []
        for i in range(len(pairs)):
            pairs_fc.append(self.relation_module_fc(feats[i].view(-1)))
        
        loss_relation = self.loss_relation(torch.sigmoid(torch.cat(pairs_fc)), pairs_targets.float())
       
        return loss_relation

    def forward_test(self, pairs, pairs_feats):
        N,_,c,h,w = pairs_feats.size()
        feats = pairs_feats.view(N,c*2,h,w)

        for conv in self.relation_module:
            feats =conv(feats)
            feats = self.relation_module_pool(feats)
        feats = F.adaptive_avg_pool2d(feats,1) 
        pairs_fc = []
        for i in range(len(pairs)):
            pairs_fc.append(torch.sigmoid(self.relation_module_fc(feats[i].view(-1))))

        return torch.cat(pairs_fc)