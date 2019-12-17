#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import cv2, random
import numpy as np
from PIL import Image

from .auto_classification import AutoAugmentation
from .auto_detection import distort_image_with_autoaugment, distort_image_with_autoaugment_mask
# from .styleaug import StyleAugmentor

from .styleaug import StyleAugmentor
import torch
from torchvision.transforms import ToTensor, ToPILImage
from imagecorruptions import corrupt
from mmdet.datasets.registry import PIPELINES
from pycocotools.coco import COCO
#------------------------------------------------------------------------------
#Style Augmentation
#------------------------------------------------------------------------------
@PIPELINES.register_module
class StyleAugmentation(object):
    def __init__(self):
        self.toTensor = ToTensor()
        self.toPIL = ToPILImage()
        self.augmentor = StyleAugmentor()
    def __call__(self, results):
        # PyTorch Tensor <-> PIL Image transforms:

        # load image:
        img = results['img']
        im_torch = self.toTensor(im).unsqueeze(0) # 1 x 3 x 256 x 256#'cuda:0' if torch.cuda.is_available() else 'cpu')
        im_torch = im_torch.to('cuda' if torch.cuda.is_available() else 'cpu')
        # create style augmentor:
         # randomize style:
        im_restyled = self.augmentor(im_torch)
        im_restyled = im_restyled.squeeze().cpu()
        im_re = self.toPIL(im_restyled)
        re_img = np.array(im_re)
        results['img'] = re_img
        return results

#------------------------------------------------------------------------------
#  ObjDetAugmentation
#------------------------------------------------------------------------------
@PIPELINES.register_module
class ObjDetAugmentation(object):
	def __init__(self, policy='v0', with_mask=True):
		assert policy in ['v0']
		self.policy = policy
		self.with_mask = with_mask
		print("[{}] Initialize with policy {}".format(self.__class__.__name__, policy))

	def __call__(self, results):
		image = results['img']
		bboxes = results['gt_bboxes']
		print("DATA KEY: \n",results.keys())
		masks = results['gt_masks']

		height, width = image.shape[:2]
		bboxes = self._normalize_bboxes(bboxes, height, width)
		if self.with_mask:
			augmented_image, augmented_bbox, augmented_mask = distort_image_with_autoaugment_mask(image, bboxes, masks, self.policy)
			results['gt_masks'] = augmented_mask
		else:
			augmented_image, augmented_bbox = distort_image_with_autoaugment(image, bboxes, self.policy)
		
		augmented_image = augmented_image.copy()
		augmented_bbox = np.array(augmented_bbox, np.float32)

		augmented_bbox = self._denormalize_bboxes(augmented_bbox, height, width)
		results['img'] = augmented_image
		results['gt_bboxes'] = augmented_bbox
		return results

	@staticmethod
	def _normalize_bboxes(bboxes, height, width):
		bboxes = bboxes[:, [1,0,3,2]]
		bboxes[:,0] /= height
		bboxes[:,1] /= width
		bboxes[:,2] /= height
		bboxes[:,3] /= width
		return bboxes

	@staticmethod
	def _denormalize_bboxes(bboxes, height, width):
		bboxes[:,0] *= height
		bboxes[:,1] *= width
		bboxes[:,2] *= height
		bboxes[:,3] *= width
		bboxes = bboxes[:,[1,0,3,2]]
		return bboxes


#------------------------------------------------------------------------------
#  ColorAutoAugmentation
#------------------------------------------------------------------------------
@PIPELINES.register_module
class ColorAutoAugmentation(object):
	"""
	Inherit from AutoAugmentation from ImageNet, but remove policies related to
	position change, just color-changing policies are remained.
	"""
	def __init__(self, fillcolor=(128, 128, 128)):
		self.policies = [
			AutoAugmentation(0.6, "solarize",  5, 0.6, "autocontrast", 5, fillcolor),
			AutoAugmentation(0.8, "equalize",  8, 0.6, "equalize",     3, fillcolor),
			AutoAugmentation(0.6, "posterize", 7, 0.6, "posterize",    6, fillcolor),
			AutoAugmentation(0.4, "equalize",  7, 0.2, "solarize",     4, fillcolor),
			AutoAugmentation(0.6, "solarize",  3, 0.6, "equalize",     7, fillcolor),
			AutoAugmentation(0.8, "posterize", 5, 1.0, "equalize",     2, fillcolor),
			AutoAugmentation(0.6, "equalize",  8, 0.4, "posterize",    6, fillcolor),
			AutoAugmentation(0.6, "equalize",  7, 0.8, "equalize",     8, fillcolor),
			AutoAugmentation(0.6, "invert",    4, 1.0, "equalize",     8, fillcolor),
			AutoAugmentation(0.6, "color",     4, 1.0, "contrast",     8, fillcolor),
			AutoAugmentation(0.8, "color",     8, 0.8, "solarize",     7, fillcolor),
			AutoAugmentation(0.4, "sharpness", 7, 0.6, "invert",       8, fillcolor),
			AutoAugmentation(0.4, "color",     4, 0.6, "equalize",     3, fillcolor),
			AutoAugmentation(0.4, "equalize",  7, 0.2, "solarize",     4, fillcolor),
			AutoAugmentation(0.6, "solarize",  5, 0.6, "autocontrast", 5, fillcolor),
			AutoAugmentation(0.6, "invert",    4, 1.0, "equalize",     8, fillcolor),
			AutoAugmentation(0.6, "color",     4, 1.0, "contrast",     8, fillcolor),
			AutoAugmentation(0.8, "equalize",  8, 0.6, "equalize",     3, fillcolor)
		]
		print("{} Initialize ColorAutoAugmentation with {} policies".format(self.__class__.__name__, len(self.policies)))

	def __call__(self, results):
		img = results['img']
		img_PIL = Image.fromarray(img.astype(np.uint8))
		policy_idx = random.randint(0, len(self.policies) - 1)
		aug_img = self.policies[policy_idx](img_PIL)
		results['img'] =  np.array(aug_img).astype(img.dtype)
		return resutls

	def __repr__(self):
		return "AutoAugment BDD Policy"


#------------------------------------------------------------------------------
#  ColorAugmentation
#------------------------------------------------------------------------------
@PIPELINES.register_module
class ColorAugmentation(object):
	"""
	A simple Augmenter for changing color of image.
	"""
	def __init__(self, gamma_range=(0.5, 1.5), noise_delta=10):
		self.gamma_range = gamma_range
		self.noise_delta = noise_delta

	def __call__(self, results):
		img = results['img']
		# Adjust brightness
		gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
		invGamma = 1.0 / gamma
		table = 255.0 * (np.arange(0, 256) / 255.0) ** invGamma
		img = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8)).astype(np.float32)

		# Add noise
		noise = np.random.normal(size=img.shape) * self.noise_delta
		img += noise
		img = np.clip(img, 0.0, 255.0)
		results['img'] = img
		return results
 
 
@PIPELINES.register_module
class Corruption(object):
	"""
	Corrupt image
	"""
	def __init__(self,
				corruption=False,
				corruption_type=['contrast','defocus_blur','gaussian_blur',
												'motion_blur','zoom_blur'], 
				corruption_severity=1):
		self.corruption = corruption
		self.corruption_type = corruption_type
		self.corruption_severity = corruption_severity

	def __call__(self, results):
		img = results['img']
		if self.corruption:
			if type(self.corruption_type) is list: 
				self.corruption_appl = self.corruption_type[np.random.choice(len(self.corruption_type),1)[0]] 
				_img = corrupt(
						img,
						severity=self.corruption_severity,
						corruption_name=self.corruption_appl)
			else:
				_img = corrupt(
						img,
						severity=self.corruption_severity,
						corruption_name=self.corruption_type)
		results['img'] = _img
		return results
