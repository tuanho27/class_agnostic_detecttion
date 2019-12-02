#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import numpy as np
from ..registry import LOSSES

try:
	from itertools import  ifilterfalse
except ImportError: # py3k
	from itertools import  filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#  Utils
#------------------------------------------------------------------------------
def isnan(x):
	return x != x

def mean(l, ignore_nan=False, empty=0):
	"""
	nanmean compatible with generators.
	"""
	l = iter(l)
	if ignore_nan:
		l = ifilterfalse(isnan, l)

	try:
		n = 1
		acc = next(l)
	except StopIteration:
		if empty == 'raise':
			raise ValueError('Empty mean')
		return empty

	for n, v in enumerate(l, 2):
		acc += v

	return acc / n

def flatten_probas(probas, labels):
	"""
	Flattens predictions in the batch
		probas: [N,C,H,W] or [N,H,W] or [N,C]
		labels: [N,C,H,W] or [N,H,W] or [N,C]
	"""
	if len(probas.shape)==3:
		probas = probas.unsqueeze(1).permute(0, 2, 3, 1).view(-1, 1)
		labels = labels.unsqueeze(1).permute(0, 2, 3, 1).view(-1, 1)

	elif len(probas.shape)==4:
		num_classes = probas.shape[1]
		probas = probas.permute(0, 2, 3, 1).view(-1, num_classes)
		labels = labels.permute(0, 2, 3, 1).view(-1, num_classes)

	return probas, labels

def lovasz_grad(gt_sorted):
	"""
	Computes gradient of the Lovasz extension w.r.t sorted errors
	See Alg. 1 in paper
	"""
	p = len(gt_sorted)
	gts = gt_sorted.sum()
	intersection = gts - gt_sorted.float().cumsum(0)
	union = gts + (1 - gt_sorted).float().cumsum(0)
	jaccard = 1. - intersection / union
	if p > 1: # cover 1-pixel case
		jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
	return jaccard

def lovasz_hinge_flat(logits, labels, margin=1.0):
	"""
	Binary Lovasz hinge loss
		logits: torch.float32, [N,C]
		labels: torch.float32, [N,C]
	"""
	# only void pixels, the gradients should be 0
	if len(labels) == 0:
		return logits.sum() * 0.0

	signs = 2.0 * labels - 1.0
	errors = margin - signs * logits
	errors_sorted, perm = torch.sort(errors, 0, descending=True)

	# Loop over classes
	losses = []
	for c in range(logits.shape[1]):
		errors_sorted_c = errors_sorted[:,c]
		perm_c = perm[:,c].data
		labels_c = labels[:,c]
		gt_sorted_c = labels_c[perm_c]

		grad_c = lovasz_grad(gt_sorted_c)
		loss_c = torch.dot(F.relu(errors_sorted_c), grad_c)
		losses.append(loss_c)

	loss = mean(losses)
	return loss


#------------------------------------------------------------------------------
#  BinaryLovaszLoss
#------------------------------------------------------------------------------
@LOSSES.register_module
class BinaryLovaszLoss(nn.Module):
	def __init__(self, per_image=False, loss_weight=1.0, margin=1.0):
		super(BinaryLovaszLoss, self).__init__()
		self.margin = margin
		self.per_image = per_image
		self.loss_weight = loss_weight

	def forward(self, logits, masks, *args, **kargs):
		"""
		logits: (torch.float32)  shape (N,C,H,W) or (N,C)
		masks: (torch.int64, torch.float32) shape (N,C,H,W) or (N,C), value in {0;1}
		"""
		masks = masks.float()
		if self.per_image:
			loss = mean(
				lovasz_hinge_flat(*flatten_probas(logit.unsqueeze(0), mask.unsqueeze(0)), margin=self.margin)
				for logit, mask in zip(logits, masks)
			)
		else:
			loss = lovasz_hinge_flat(*flatten_probas(logits, masks), margin=self.margin)

		return loss * self.loss_weight
