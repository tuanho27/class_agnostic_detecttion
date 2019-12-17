import os, cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import init_detector, inference_detector, show_result, LoadImage


def get_data(img, cfg, device):
	# build the data pipeline
	test_pipeline = [LoadImage()] + cfg.test_pipeline[1:]
	test_pipeline = Compose(test_pipeline)
	# prepare data
	data = dict(img=img)
	data = test_pipeline(data)
	data = scatter(collate([data], samples_per_gpu=1), [device])[0]
	return data


config = "ccdetection/configs/polarmask/polar_b1_semseg.py"
checkpoint = "work_dirs/polar-B1-FPN-yolact_1image/epoch_29.pth"
img_file = "work_dirs/000000397133.jpg"
out_file = "work_dirs/000000397133_out.png"


model = init_detector(config, checkpoint=checkpoint, device='cuda')
# data = get_data(img_file, model.cfg, next(model.parameters()).device)
# print(data.keys())
# with torch.no_grad():
# 	img = data['img'][0]
# 	ori_shape = data['img_meta'][0][0]['img_shape']

# 	x = model.extract_feat(img)
# 	outs = model.bbox_head(x)
# 	mask_pred = model.semseg_head(x)
# 	mask_pred = model.semseg_head.get_seg_masks(mask_pred, ori_shape, scale_factor=1.0, rescale=True, threshold=0.5)
# 	mask_pred = mask_pred[0,0].astype('uint8')
# 	print('mask_pred', mask_pred.shape, mask_pred.min(), mask_pred.max())

results = inference_detector(model, img_file)

# show the results
show_result(img_file, results, model.CLASSES,show=False, out_file=out_file)

# image = cv2.imread(img_file)[...,::-1]
# image = cv2.resize(image, (mask_pred.shape[1], mask_pred.shape[0]), interpolation=cv2.INTER_LINEAR)
# mask = np.zeros_like(image)
# mask[mask_pred==1,...] = (0,0,255)
# image = cv2.add(image, mask)
# cv2.imwrite(out_file, image[...,::-1])
# cv2.imwrite(out_file, 255*mask_pred)
