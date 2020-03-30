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
    colors=['blue','cyan', 'green'] #, 'red', 'yellow', 'magenta']
    colors = [mmcv.color_val(c) for c in colors]
    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    size = config.data.test.pipeline[2]['img_scale']

    if args.img_folder is None:
        print("\nCould not find the image folder for infering, please check!!!")
        return 0
    img_files = sorted(glob.glob(f'{args.img_folder}/*.png')) #glob.glob(f'{args.img_folder}/*.png') #

    #############################################
    ### Test all dataset in data folder
    # out_folder= './testing/outputs/{}'.format(args.img_folder.split("/")[-1])
    # if not os.path.isdir(out_folder):
    #     os.mkdir(out_folder)

    # model = init_detector(args.config, args.checkpoint, device='cuda')
    # print("Start infer model !!!\n")

    # ### test_folder
    # count = 0
    # for frame, image in enumerate(img_files):
    #     count+=1
    #     results = inference_pair_detector(model, [img_files[frame],img_files[frame+1]])
    #     img0 = mmcv.imread(img_files[frame])    
    #     img0 = mmcv.imresize(img0, size, return_scale=True)[0]
    #     # img1 = mmcv.imflip(img0) ## just flip to write, the flip is done by load data function
    #     img1 = mmcv.imread(img_files[frame+1])    
    #     img1 = mmcv.imresize(img1, size, return_scale=True)[0]
    #     # show the results
    #     for i, out in enumerate(results):
    #         bbox_int_0 = out[0].cpu().numpy().astype(np.int32)
    #         bbox_int_1 = out[1].cpu().numpy().astype(np.int32)
    #         cl = random.randint(1,len(colors)-1)
    #         cv2.rectangle(img0, (bbox_int_0[0], bbox_int_0[1]), (bbox_int_0[2], bbox_int_0[3]), colors[cl], thickness=2)
    #         cv2.rectangle(img1, (bbox_int_1[0], bbox_int_1[1]), (bbox_int_1[2], bbox_int_1[3]), colors[cl], thickness=2)

    #     img = np.concatenate((img0, img1), axis=1) 
    #     cv2.imwrite("./{}/{}_{}.jpg".format(out_folder,img_files[frame].split(".")[0].split("/")[-1], frame), img)
    ##     if count == 2:
    ##         break

    ###############################################
    ### Test single images 
    #cat 
    # img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/000215.jpg"
    # img1_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/000122.jpg"

    #sheep 
    img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/006678.jpg"
    img1_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/002209.jpg"
    # img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/007230.jpg" ## two sheeps

    #horse 
    # img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/001013.jpg"
    # img1_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/000056.jpg"
    
    #cat,horse + person
    # img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/001173.jpg"
    # img1_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/001769.jpg" 

    ##### False cases ######
    #horses + person 
    # img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007/JPEGImages/003889.jpg"
    # img1_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/001452.jpg" 

    #person + animals
    # img0_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/001769.jpg" 
    # img1_file = "/home/member/Workspace/dataset/VOC/VOCdevkit/VOC2007TEST/JPEGImages/001914.jpg" 


    #toyota images
    # img0_file  = img_files[1]
    # img1_file = img_files[200]

    model = init_detector(args.config, args.checkpoint, device='cuda')
    print("Start infer model !!!\n")
    results, stage2_results, scores = inference_pair_detector(model, [img0_file,img1_file])

    img0 = mmcv.imread(img0_file)    
    img0 = mmcv.imresize(img0, size, return_scale=True)[0]

    img1 = mmcv.imread(img1_file)    
    img1 = mmcv.imresize(img1, size, return_scale=True)[0]
    # img1 = mmcv.imflip(img0) #(just to display)

    # show the results
    for i, out in enumerate(results):
        bbox_int_0 = out[0].cpu().numpy().astype(np.int32)
        bbox_int_1 = out[1].cpu().numpy().astype(np.int32)

        cl = random.randint(1,len(colors)-1)
        cv2.rectangle(img0, (bbox_int_0[0], bbox_int_0[1]), (bbox_int_0[2], bbox_int_0[3]), colors[cl], thickness=2)
        cv2.rectangle(img1, (bbox_int_1[0], bbox_int_1[1]), (bbox_int_1[2], bbox_int_1[3]), colors[cl], thickness=2)

    img = np.concatenate((img0, img1), axis=1)

    for out, score in zip(results, scores):    
        centres_0=(0.5*(out[0][0] + out[0][2]), 0.5*(out[0][1] + out[0][3])) 
        centres_1=(0.5*(out[1][0] + out[1][2]) + size[0], 0.5*(out[1][1] + out[1][3])) 

        # import ipdb; ipdb.set_trace()
        cv2.line(img, (int(centres_0[0]),int(centres_0[1])), (int(centres_1[0]),int(centres_1[1])), (0, 64, 255), 2)
        cv2.putText(img,'Match score: %f'%score, (int((centres_0[0]+centres_1[0])*0.5),int((centres_0[1]+centres_1[1])*0.5)), 
                                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (256,64,0), 2)
    cv2.imwrite("./testing/False_sheep_output_pair_test.jpg", img)
    
    # show_result(img0, stage2_results[0], model.CLASSES,0.3, show=False,out_file="./testing/output_0.png")
    # show_result(img1, stage2_results[1], model.CLASSES,0.3, show=False,out_file="./testing/output_1.png")


if __name__ == '__main__':
    main()
