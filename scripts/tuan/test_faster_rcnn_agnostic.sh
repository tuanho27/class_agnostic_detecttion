#!/usr/bin/env bash

## COCO model
# CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco14.py'
# WORK_DIR='./work_dirs/faster_rcnn_r50_fpn_1x_coco14_update_data'

## VOC model
CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
WORK_DIR='./work_dirs/faster_rcnn_r50_fpn_1x_voc0712_update_data'

RESULT_FILE='./result.pkl'
# IMG_FOLDER='/home/member/Workspace/dataset/from_toyota/20200303-20200305T040634Z-158/20200303/20200227_realsense_2/20200227_123446'
IMG_FOLDER='/home/member/Workspace/dataset/from_toyota/20200303-20200305T040634Z-158/20200303/20200227_realsense_2/20200227_120311'

th=15
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth" 

GPUS=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=${PYTHON:-"python"}

## for inferencce images or videos
python mmdetection/tools/infer_pair_images.py --config ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} \
                                            --img_folder $IMG_FOLDER

## for evaluation
# python    mmdetection/tools/test_pair_eval.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${RESULT_FILE} --eval bbox

