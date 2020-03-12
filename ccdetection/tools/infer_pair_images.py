import os, cv2
import numpy as np
import argparse
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.apis.inference import init_detector, inference_detector, inference_pair_detector, show_result, LoadImage
import mmcv
import random
import glob

def get_data(img, cfg, device):
	# build the data pipeline
	test_pipeline = [LoadImage()] + cfg.test_pipeline[1:]
	test_pipeline = Compose(test_pipeline)
	# prepare data
	data = dict(img=img)
	data = test_pipeline(data)
	data = scatter(collate([data], samples_per_gpu=1), [device])[0]
	return data

def parse_args():
	parser = argparse.ArgumentParser(description='test detector')
	parser.add_argument('--config', help='test config file path')
	parser.add_argument('--checkpoint', help='checkpoint file')
	parser.add_argument('--img_file',nargs="+", default=[], type=str, help='Image path to infering')
	parser.add_argument('--img_folder', help='folder of test images')

	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	colors=['green', 'red', 'blue', 'yellow', 'magenta','cyan', 'white']
	colors = [mmcv.color_val(c) for c in colors]

	if args.img_folder is None:
		print("\nCould not find the image folder for infering, please check!!!")
		return 0
	img_files = glob.glob(f'{args.img_folder}/*.png')

	out_folder= './testing/outputs/{}'.format(args.img_folder.split("/")[-1])
	if not os.path.isdir(out_folder):
		os.mkdir(out_folder)

	model = init_detector(args.config, args.checkpoint, device='cuda')
	print("Start infer model !!!\n")

	### test_folder
	for frame, image in enumerate(img_files):
		results = inference_pair_detector(model, [img_files[frame],img_files[frame+1]])
		img0 = mmcv.imread(img_files[frame])    
		img0 = mmcv.imresize(img0, (900, 600), return_scale=True)[0]

		# img1 = mmcv.imflip(img0)

		img1 = mmcv.imread(img_files[frame+1])    
		img1 = mmcv.imresize(img1, (900, 600), return_scale=True)[0]

		# show the results
		for i, out in enumerate(results):
			bbox_int_0 = out[0].cpu().numpy().astype(np.int32)
			bbox_int_1 = out[1].cpu().numpy().astype(np.int32)

			cl = random.randint(1,len(colors)-1)
			cv2.rectangle(img0, (bbox_int_0[0], bbox_int_0[1]), (bbox_int_0[2], bbox_int_0[3]), colors[cl], thickness=2)
			cv2.rectangle(img1, (bbox_int_1[0], bbox_int_1[1]), (bbox_int_1[2], bbox_int_1[3]), colors[cl], thickness=2)

		img = np.concatenate((img0, img1), axis=1) 
		cv2.imwrite("./{}/{}_{}.jpg".format(out_folder,img0_file.split(".")[0].split("/")[-1], frame), img)

	### Test single images
	# img0_file = "testing/2007_003889.jpg"
	# img1_file = "testing/2007_005114.jpg"
	
	# img0_file = "testing/2007_004193.jpg"
	# img1_file = "testing/2007_005902.jpg"

	# img0_file = "testing/20200227_021324_050708294.png"
	# img1_file = "testing/20200227_021325_193340063.png"

	# img0_file = "testing/20200227_080058_981132984.png"
	# img1_file = "testing/20200227_080102_946602821.png"

	# model = init_detector(args.config, args.checkpoint, device='cuda')
	# print("Start infer model !!!\n")
	# results = inference_pair_detector(model, [img0_file,img0_file])
	
	# img0 = mmcv.imread(img0_file)    
	# img0 = mmcv.imresize(img0, (900, 600), return_scale=True)[0]

	# # img1 = mmcv.imread(img1_file)    
	# # img1 = mmcv.imresize(img1, (900, 600), return_scale=True)[0]
	# img1 = mmcv.imflip(img0)

	# # show the results
	# for i, out in enumerate(results):
	# 	bbox_int_0 = out[0].cpu().numpy().astype(np.int32)
	# 	bbox_int_1 = out[1].cpu().numpy().astype(np.int32)

	# 	cl = random.randint(1,len(colors)-1)
	# 	cv2.rectangle(img0, (bbox_int_0[0], bbox_int_0[1]), (bbox_int_0[2], bbox_int_0[3]), colors[cl], thickness=2)
	# 	cv2.rectangle(img1, (bbox_int_1[0], bbox_int_1[1]), (bbox_int_1[2], bbox_int_1[3]), colors[cl], thickness=2)
	# cv2.imwrite("./testing/outputs/{}.jpg".format(img0_file.split(".")[0].split("/")[-1]), img0)
	# cv2.imwrite("./testing/outputs/{}_flip.jpg".format(img0_file.split(".")[0].split("/")[-1]), img1)


if __name__ == '__main__':
    main()
