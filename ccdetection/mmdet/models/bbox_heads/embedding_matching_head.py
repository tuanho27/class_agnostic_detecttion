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
    def __init__(self, margin=1e-1, 
                       lamb = 2,
                       in_channels=256,
                       feat_channels=128,
                       cls_out_channels = 1, # background and foreground classes
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
        self.loss_siamese = build_loss(loss_siamese)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.siamese_embedding = nn.Sequential(
                                 nn.Linear(7*7*in_channels,512),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512,self.feat_channels))
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


    def forward(self, pairs, pairs_targets, pairs_feats):#, pairs_bbox_target, pairs_bbox_target_weight):
        start = time()
        dist_pairs = []
        ## siamese loss
        for i in range(len(pairs)):
            pairs_fc = [self.siamese_embedding(pairs_feats[i][j].view(-1)) for j in range(2)]
            if self.dist_mode == 'cosine':
                if pairs_targets[i] == True:
                    dist_pairs.append(torch.log(self.dist(pairs_fc[0],pairs_fc[1])))
                else:
                    dist_pairs.append(torch.log(1 - self.dist(pairs_fc[0],pairs_fc[1]) + self.margin))
            else:
                if pairs_targets[i] == True:
                    dist_pairs.append(torch.log((pairs_fc[0] - pairs_fc[1]).pow(2).sum(1)))
                else:
                    dist_pairs.append(torch.log(1- (pairs_fc[0] - pairs_fc[1] + self.margin).pow(2).sum(1)))
        dist_pairs = torch.stack(dist_pairs)
        loss_siamese = self.loss_siamese(dist_pairs[:,None], pairs_targets.long(),avg_factor=len(pairs))

        ## refine loss
        # feats = self.refine_rpn_conv(torch.cat([feat for feat in pairs_feats]))
        # feats = F.relu(feats, inplace=True)
        # feats = F.adaptive_avg_pool2d(feats,1)

        # refine_cls_score = self.refine_rpn_cls(feats)
        # refine_bbox_pred = self.refine_rpn_reg(feats)

        ## classification loss
        # cls_score = refine_cls_score.reshape(-1, self.cls_out_channels)
        # loss_cls = self.loss_cls(cls_score, pairs_targets.repeat_interleave(2).long(), avg_factor=len(pairs))

        ## regression loss
        # pairs_bbox_target = torch.cat([_ for _ in pairs_bbox_target]) #pairs_bbox_target.reshape(-1, 4)
        # pairs_bbox_target_weight = pairs_bbox_target_weight.reshape(-1, 4)
        # bbox_pred = refine_bbox_pred.reshape(-1, 4)
        # loss_bbox = self.loss_bbox(bbox_pred, pairs_bbox_target, pairs_bbox_target_weight, avg_factor=len(pairs))

        ## losses = loss_cls + loss_bbox + self.lamb*loss_siamese
        # losses = loss_bbox + self.lamb*loss_siamese

        end = time() - start
        # print("Time for siamese", end) 
        return loss_siamese #dict(loss_siamese=loss_siamese)

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
                       num_conv = 3,
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

    def forward(self, pairs, pairs_targets, pairs_feats): #, pairs_bbox_target, pairs_bbox_target_weight):
        start = time()
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

        ## refine loss
        # feats = self.refine_rpn_conv(torch.cat([feat for feat in pairs_feats]))
        # feats = F.relu(feats, inplace=True)
        # feats = F.adaptive_avg_pool2d(feats,1)

        # refine_cls_score = self.refine_rpn_cls(feats)
        # refine_bbox_pred = self.refine_rpn_reg(feats)

        ## classification loss
        # cls_score = refine_cls_score.reshape(-1, self.cls_out_channels)
        # loss_cls = self.loss_cls(cls_score, pairs_targets.repeat_interleave(2).long(), avg_factor=len(pairs))

        ## regression loss
        # pairs_bbox_target = torch.cat([_ for _ in pairs_bbox_target]) #pairs_bbox_target.reshape(-1, 4)
        # pairs_bbox_target_weight = pairs_bbox_target_weight.reshape(-1, 4) 
        # bbox_pred = refine_bbox_pred.reshape(-1, 4)
        # loss_bbox = self.loss_bbox(bbox_pred, pairs_bbox_target, pairs_bbox_target_weight, avg_factor=len(pairs))

        ## losses = loss_cls + loss_bbox+ self.lamb*loss_relation
        # losses = loss_bbox + self.lamb*loss_relation

        # print("Time for relation", time()-start)
        return loss_relation #dict(loss_relation=loss_relation)


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