import json_tricks as json
import numpy as np
import os
import cv2

img_prefix= '/home/chuong/Workspace/dataset/mpii/images/'
ann_prefix= '/home/chuong/Workspace/dataset/mpii/annot/'

for subset in ['train','valid','trainval']:
	# create train/val split
	ann_file = os.path.join(ann_prefix,f'{subset}.json')
	with open(ann_file) as anno_file:
		anno = json.load(anno_file)

	img_info=dict()
	for idx,a in enumerate(anno):
		#Parse notation
		image_name = a['image']
		print(subset,idx)
		img_id=image_name.replace('.jpg','')

		c = np.array(a['center'], dtype=np.float)
		s = np.array([a['scale'], a['scale']], dtype=np.float)
		# Adjust center/scale slightly to avoid cropping limbs
		if c[0] != -1:
			c[1] = c[1] + 15 * s[1]
			s = s * 1.25
		# MPII uses matlab format, index is based 1,
		# we should first convert to 0-based index
		c = c - 1
		joints = np.array(a['joints'])
		joints[:, 0:2] = joints[:, 0:2] - 1
		joints_vis = np.array(a['joints_vis'])

		# save into dict:
		if img_id in img_info:
			img_info[img_id]['center'].append(c)
			img_info[img_id]['scale'].append(s)
			img_info[img_id]['joints'].append(joints)
			img_info[img_id]['joints_vis'].append(joints_vis)
		else:
			filename=os.path.join(img_prefix,image_name)
			img=cv2.imread(filename)
			h,w,_=img.shape
			item=dict(filename=image_name,width=w,height=h,center=[c],scale=[s],joints=[joints],joints_vis=[joints_vis])
			img_info[img_id]=item

	img_info=[v for k,v in img_info.items()]
	outfile = os.path.join(ann_prefix,f'{subset}_multi.json')
	with open(outfile, 'w') as f:
		f.write(json.dumps(img_info, indent=4))