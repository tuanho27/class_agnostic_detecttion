# %%
# %load_ext autoreload
# %autoreload 2
import os
import numpy as np
import pandas as pd
import pickle
import cv2

# Segmentation related
def mask2rle(img):
	'''
		img: numpy array, 1 - mask, 0 - background
		Returns run length as string formated
	'''
	pixels= img.T.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0]
	# import pdb; pdb.set_trace()
	runs[1::2] -= runs[::2]
	runs[1::2] +=1
	return ' '.join(str(x) for x in runs)

def rle2mask(rle, img_shape=(256,1600)):
	# Parse string
	array = np.asarray([int(v) for v in rle.split()])
	start = array[0::2]
	length = array[1::2]

	# Create Mask
	mask = np.zeros(img_shape[0]*img_shape[1], dtype=np.uint8)
	for (start,length) in zip(start,length):
		mask[start:(start+length-1)]=1
	mask = mask.reshape(img_shape[0],img_shape[1], order='F')
	return mask

def bounding_box(img):
	# return max and min of a mask to draw bounding box
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]

	return rmin, rmax, cmin, cmax

CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']
CLASSES_IDX = {k:i for i,k in enumerate(CLASSES,start=1)}

def create_cloud_dataset(csv_file,max_imgs=None,out_folder=None, min_area = 400):
	# load full data and label no mask as -1
	train_df = pd.read_csv(csv_file).fillna(-1)
	# image id and class id are two seperate entities and it makes it easier to split them up in two columns
	train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
	train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
	# lets create a dict with class id and encoded pixels and group all the defaults per image
	train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)
	grouped_EncodedPixels = train_df.groupby('ImageId')['Label_EncodedPixels'].apply(list)

	# Convert annotation into bboxes and instance_mask
	l = len(grouped_EncodedPixels)
	lMax = l if max_imgs is None else min(l,max_imgs)

	img_infos=[]
	ann_infos=[]
	h,w=1400,2100
	for id,(filename,label_mask) in enumerate(grouped_EncodedPixels.iteritems()):
		if id >lMax:
			break

		print(filename)
		labels=[]
		bboxes=[]
		masks =[]
		for label, mask in label_mask:
			if mask!=-1:
				mask_decoded = rle2mask(mask, (h,w))
				#Extract components and its bounding box
				nComponent, label_components, stats, centroids= cv2.connectedComponentsWithStats(mask_decoded)
				if nComponent >2:
					valid_component = stats[:,cv2.cv2.CC_STAT_AREA] > min_area
					valid_component[0] = False
					#ignore the first component which is the ground
				else:
					#ignore the first component which is the ground
					valid_component = [False, True]

				# Work on each instance in the current class
				left = stats[valid_component, cv2.CC_STAT_LEFT][:,np.newaxis]
				top = stats[valid_component, cv2.CC_STAT_TOP][:,np.newaxis]
				right  = left + stats[valid_component, cv2.CC_STAT_WIDTH][:,np.newaxis]
				bottom = top + stats[valid_component, cv2.CC_STAT_HEIGHT][:,np.newaxis]
				instance_bboxes = np.concatenate((left,top,right,bottom),axis=1)

				instance_rle =[]
				for ith_component, is_valid in enumerate(valid_component):
					if is_valid:
						mask_i =  np.uint8((label_components==ith_component))
						rle_i  = mask2rle(mask_i)
						instance_rle.append(rle_i)
				instance_labels=np.array([CLASSES_IDX[label]]*len(instance_rle))

				labels.append(instance_labels)
				bboxes.append(instance_bboxes)
				masks+=instance_rle

		labels = np.concatenate(tuple(labels))
		bboxes = np.concatenate(tuple(bboxes))

		img_infos.append(dict(filename=filename, height=h,width=w))
		ann_infos.append(dict(bboxes=bboxes,labels=labels,masks=masks))

	if out_folder is None:
		out_folder = os.path.dirname(csv_file)
	ann_output = os.path.join(out_folder,'train_ann.pickle')
	with open(ann_output, 'wb') as f:
		pickle.dump(dict(img_infos=img_infos,ann_infos=ann_infos),f)

if __name__ == '__main__':
	#root_folder = '/home/chuong/Workspace/dataset/cloud/'
	root_folder = '/home/member/Workspace/chuong/dataset/cloud/'
	csv_file = os.path.join(root_folder,'train.csv')
	create_cloud_dataset(csv_file=csv_file, max_imgs=None, min_area=1600)
