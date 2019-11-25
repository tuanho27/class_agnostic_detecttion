#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..registry import LOSSES


#------------------------------------------------------------------------------
#  FocalKLLoss
#------------------------------------------------------------------------------
@LOSSES.register_module
class FocalKLLoss(nn.Module):
	def __init__(self, use_sigmoid=True, loss_weight=1.0, gamma=1.0):
		super(FocalKLLoss, self).__init__()
		assert use_sigmoid is True, 'Only sigmoid focaloss supported now.'
		self.use_sigmoid = use_sigmoid
		self.loss_weight = loss_weight
		self.gamma = gamma

	def forward(self, input, target, label_weights, avg_factor):
		'''
		input  : logit, shape (N, num_classes)
		target : logit, shape (N, num_classes)
		label_weights: shape (N, )
		'''
		logpx = F.logsigmoid(input)
		log1minuspx = -input + logpx
		px = logpx.exp()
		py = target.sigmoid()
		abs_subtraction_gamma = torch.abs(py-px)**self.gamma

		loss1 = F.kl_div(logpx, py, reduction='none')
		loss2 = F.kl_div(log1minuspx, 1-py, reduction='none')

		loss =  abs_subtraction_gamma * (loss1 + loss2)
		loss = torch.sum(loss * label_weights[:,None]) / avg_factor
		return loss * self.loss_weight
