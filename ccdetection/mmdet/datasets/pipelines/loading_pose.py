import os.path as osp
import warnings

import mmcv
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadPoseAnnotations(object):
	def __init__(self,
				 with_joints =True,
				 with_heatmap=True,
				 with_affinity=False,
				 with_PIF=False,
				 with_PAF=False,
				 feat_stride=4,
				 gauss_sigma=3,
				 skip_img_without_anno=True):
		assert with_joints
		assert with_heatmap
		self.sigma=gauss_sigma
		self.feat_stride=feat_stride

		self.with_joints = with_joints
		self.with_heatmap = with_heatmap
		self.with_affinity = with_affinity
		self.with_PIF = with_PIF
		self.with_PAF = with_PAF
		self.skip_img_without_anno = skip_img_without_anno

	def _load_joints(self, results):
		ann_info = results['ann_info']
		import pdb; pdb.set_trace()
		n_object = len(ann_info['joints'])
		results['gt_joints'] = np.stack([joint/self.feat_stride for joint in ann_info['joints']],axis=0).reshape(n_object,-1)
		results['gt_joints_vis'] = np.stack(ann_info['joints_vis'],axis=0)
		results['point_fields'].append('gt_joints')
		return results

	def _load_heatmap(self, results):
		H = results['img_info']['height']
		W = results['img_info']['width']
		heatmap_size=(W//self.feat_stride,H//self.feat_stride)

		ann_info = results['ann_info']
		multi_joints = ann_info['joints']
		multi_joints_vis = ann_info['joints_vis']
		num_joints = multi_joints[0].shape[0]

		tmp_size = self.sigma * 3
		target = np.zeros((num_joints,heatmap_size[1],heatmap_size[0]),
								dtype=np.float32)

		for joints, joints_vis in zip(multi_joints, multi_joints_vis):
			target_weight = np.ones((num_joints, 1), dtype=np.float32)
			target_weight[:, 0] = joints_vis[:,0]

			for joint_id in range(num_joints):
				mu_x = int(joints[joint_id][0] / self.feat_stride + 0.5)
				mu_y = int(joints[joint_id][1] / self.feat_stride + 0.5)
				# Check that any part of the gaussian is in-bounds
				ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
				br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
				if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
						or br[0] < 0 or br[1] < 0:
					# If not, just return the image as is
					target_weight[joint_id] = 0
					continue

				# # Generate gaussian
				size = 2 * tmp_size + 1
				x = np.arange(0, size, 1, np.float32)
				y = x[:, np.newaxis]
				x0 = y0 = size // 2
				# The gaussian is not normalized, we want the center value to equal 1
				g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

				# Usable gaussian range
				g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
				g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
				# Image range
				img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
				img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

				v = target_weight[joint_id]
				if v > 0.5:
					temp = target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]]
					target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
						np.maximum(g[g_y[0]:g_y[1], g_x[0]:g_x[1]],temp)

		results['gt_heatmap']=target
		results['mask_fields'].append('gt_heatmap')
		return results

	def __call__(self, results):
		if self.with_joints:
			results = self._load_joints(results)
		if self.with_heatmap:
			results = self._load_heatmap(results)
		if self.with_affinity:
			raise NotImplementedError
		if self.with_PIF:
			raise NotImplementedError
		if self.with_PAF:
			raise NotImplementedError

		return results

	def __repr__(self):
		repr_str = self.__class__.__name__
		repr_str += ('(with_joints={}, with_heatmap={}, with_affinity={},'
					 ' with_PIF={})',' with_PAF={})').format(self.with_joints, self.with_heatmap,
						self.with_affinity, self.with_PIF, self.with_PAF)
		return repr_str


