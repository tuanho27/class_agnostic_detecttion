#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import random
import numpy as np
from PIL import ImageEnhance, ImageOps


#------------------------------------------------------------------------------
#  AutoAugment: Learning Augmentation Policies from Data
# https://arxiv.org/abs/1805.09501
#------------------------------------------------------------------------------

class AutoAugmentation(object):
	def __init__(self, p1, operation1, magnitude_idx1,
					   p2, operation2, magnitude_idx2,
					   fillcolor=(128, 128, 128)):
		ranges = {
			"color": np.linspace(0.0, 0.9, 10),
			"posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
			"solarize": np.linspace(256, 0, 10),
			"contrast": np.linspace(0.0, 0.9, 10),
			"sharpness": np.linspace(0.0, 0.9, 10),
			"brightness": np.linspace(0.0, 0.9, 10),
			"autocontrast": [0] * 10,
			"equalize": [0] * 10,
			"invert": [0] * 10
		}
		self.p1 = p1
		self.operation1 = getattr(self, '_%s'%(operation1))
		self.magnitude1 = ranges[operation1][magnitude_idx1]
		self.p2 = p2
		self.operation2 = getattr(self, '_%s'%(operation2))
		self.magnitude2 = ranges[operation2][magnitude_idx2]

	def __call__(self, img):
		if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
		if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
		return img

	def _color(self, img, magnitude):
		return ImageEnhance.Color(img).enhance(1+magnitude*random.choice([-1,1]))

	def _posterize(self, img, magnitude):
		return ImageOps.posterize(img, magnitude)

	def _solarize(self, img, magnitude):
		return ImageOps.solarize(img, magnitude)

	def _contrast(self, img, magnitude):
		return ImageEnhance.Contrast(img).enhance(1+magnitude*random.choice([-1,1]))

	def _sharpness(self, img, magnitude):
		return ImageEnhance.Sharpness(img).enhance(1+magnitude*random.choice([-1,1]))

	def _brightness(self, img, magnitude):
		return ImageEnhance.Brightness(img).enhance(1+magnitude*random.choice([-1,1]))

	def _autocontrast(self, img, magnitude):
		return ImageOps.autocontrast(img)

	def _equalize(self, img, magnitude):
		return ImageOps.equalize(img)

	def _invert(self, img, magnitude):
		return ImageOps.invert(img)
