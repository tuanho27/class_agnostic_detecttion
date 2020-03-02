import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule
import math
from scipy.spatial import distance as dist

class SiameseMatching(nn.Module):
    """
    Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Positive and negative pairs
    """

    def __init__(self, margin=None, mode='cosine'):
        super(SiameseMatching, self).__init__()
        self.margin = margin
        self.mode = mode
        self.siamese_embedding = nn.Sequential(
                                 nn.Linear(7*7*256,4096),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4096,512))
        self.dist = nn.CosineSimilarity(dim=0, eps=1e-7) 

    def forward(self, pairs_positive, pairs_positive_feats_0,pairs_positive_feats_1,
                        pairs_negative, pairs_negative_feats_0,pairs_negative_feats_1):

        loss_positive = 0
        for i in range(pairs_positive.size()[0]):
            pairs_positive_fc_0 = self.siamese_embedding(pairs_positive_feats_0[i].view(-1))
            pairs_positive_fc_1 = self.siamese_embedding(pairs_positive_feats_1[i].view(-1))
            if self.mode == 'cosine':
                loss_positive += torch.log(self.dist(pairs_positive_fc_0,pairs_positive_fc_1))
            else:
                loss_positive += torch.log(dist.euclidean(pairs_positive_fc_0,pairs_positive_fc_1))
                
        loss_negative = 0
        for i in range(pairs_negative.size()[0]):
            pairs_negative_fc_0 = self.siamese_embedding(pairs_negative_feats_0[i].view(-1))
            pairs_negative_fc_1 = self.siamese_embedding(pairs_negative_feats_1[i].view(-1))
            if self.mode == 'cosine':
                loss_negative += torch.log(1 - self.dist(pairs_negative_fc_0,pairs_negative_fc_1))
            else:
                loss_negative += torch.log(1 - dist.euclidean(pairs_negative_fc_0,pairs_negative_fc_1))
                
        loss = loss_positive/pairs_positive.size()[0] + loss_negative/pairs_negative.size()[0]
        return dict(loss_siamese=loss)


class RelationMatching(nn.Module):
    """
    Relation Matching loss
    Takes a batch of embeddings and corresponding labels.
    Positive and negative pairs
    """

    def __init__(self):
        super(RelationMatching, self).__init__()
        self.relation_module = nn.ModuleList()
        for i in range(3):
            in_channel = 512 if i == 0 else 256
            self.relation_module.append(
                ConvModule(
                    in_channel,
                    256,
                    kernel_size=(3,3),
                    padding=(2,2),
                    conv_cfg=None,
                    norm_cfg=None))
        self.relation_module_pool = nn.MaxPool2d(kernel_size=(2,2))
        self.relation_module_fc = nn.Sequential(
                                    nn.Linear(2*2*256,128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128,1))
        # self.loss = nn.Cross
    def forward(self, pairs_positive, pairs_positive_feats_0,pairs_positive_feats_1,
                            pairs_negative, pairs_negative_feats_0,pairs_negative_feats_1):
        
        loss_positive = 0        
        pairs_positive_cat = torch.cat((pairs_positive_feats_0,pairs_positive_feats_1),dim=1)
        for i in range(3):
            pairs_positive_cat = self.relation_module[i](pairs_positive_cat)
            pairs_positive_cat = self.relation_module_pool(pairs_positive_cat)
        for i in range(pairs_positive.size()[0]):
            loss_positive += (torch.sigmoid(self.relation_module_fc(pairs_positive_cat[i].view(-1))) - 1)**2


        pairs_negative_cat = torch.cat((pairs_negative_feats_0,pairs_negative_feats_1),dim=1)
        loss_negative = 0
        for i in range(3):
            pairs_negative_cat = self.relation_module[i](pairs_negative_cat)
            pairs_negative_cat = self.relation_module_pool(pairs_negative_cat)
        for i in range(pairs_negative.size()[0]):
            loss_negative += (torch.sigmoid(self.relation_module_fc(pairs_negative_cat[i].view(-1))) - 1)**2

        # for i in range(pairs_positive.size()[0]):
        #     pairs_positive_cat = torch.cat((pairs_positive_feats_0[i][None,:],pairs_positive_feats_1[i][None,:]),dim=1)
        #     for i in range(3):
        #         pairs_positive_cat = self.relation_module[i](pairs_positive_cat)
        #         pairs_positive_cat = self.relation_module_pool(pairs_positive_cat)
        #     loss_positive += (torch.sigmoid(self.relation_module_fc(pairs_positive_cat.view(-1))) - 1)**2

        # loss_negative = 0
        # for i in range(pairs_negative.size()[0]):
        #     pairs_negative_cat = torch.cat((pairs_negative_feats_0[i][None,:],pairs_negative_feats_1[i][None,:]),dim=1)
        #     for i in range(3):
        #         pairs_negative_cat = self.relation_module[i](pairs_negative_cat)
        #         pairs_negative_cat = self.relation_module_pool(pairs_negative_cat)
        #     loss_negative += (torch.sigmoid(self.relation_module_fc(pairs_negative_cat.view(-1))))**2
                
        loss = loss_negative/pairs_negative.size()[0] + loss_negative/pairs_negative.size()[0]
        return dict(loss_relation=loss)
