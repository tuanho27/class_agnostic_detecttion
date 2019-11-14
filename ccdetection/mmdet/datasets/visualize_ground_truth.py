import mmcv
import matplotlib.pyplot as plt
import numpy as np

def show_gt(img,
				result,
				class_names,
				wait_time=0,
				show=True,
				out_file=None, color='red',
				thickness=2, font_scale=1):
	"""Visualize the detection results on the image.

	Args:
		img (str or np.ndarray): Image filename or loaded image.
		result (tuple[list] or list): The detection result, can be either
			(bbox, segm) or just bbox.
		class_names (list[str] or tuple[str]): A list of class names.
		wait_time (int): Value of waitKey param.
		show (bool, optional): Whether to show the image with opencv or not.
		out_file (str, optional): If specified, the visualization result will
			be written to the out file instead of shown in a window.

	Returns:
		np.ndarray or None: If neither `show` nor `out_file` is specified, the
			visualized image is returned, otherwise None is returned.
	"""
	assert isinstance(class_names, (tuple, list))
	img = mmcv.imread(img)
	img = img.copy()
	if len(result)==3:
		gt_labels,gt_bboxes, masks = result
	else:
		gt_labels,gt_bboxes = result
		masks= None

	labels = gt_labels.tolist()
	# draw segmentation masks
	if masks is not None:
		for i in range(len(labels)):
			color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
			mask_i = masks[i].astype(np.bool)
			img[mask_i,:] = img[mask_i,:] * 0.5 + color_mask * 0.5

	# draw bounding boxes
	mmcv.imshow_det_bboxes(
		img,
		gt_bboxes,
		gt_labels,
		class_names=class_names,
		score_thr=0,
		show=show,
		bbox_color=color,
		text_color=color,
		wait_time=wait_time,
		out_file=out_file,
		thickness=thickness, font_scale=font_scale)
	if not (show or out_file):
		return img


def visualize_gt(img_input, gt_labels,gt_bboxes,gt_masks, class_names, fig_size=(15,10)):
	result = (gt_labels,gt_bboxes,gt_masks)
	img = show_gt(
		img_input, result, class_names, show=False)
	plt.figure(figsize=fig_size)
	plt.imshow(mmcv.bgr2rgb(img))