#!/usr/bin/env bash

CONFIG_FILE="/home/member/Workspace/xuanphu/Work/ccdetpose/ccdetection/configs/maskscoring/ms_rcnn_r50_caffe_fpn_1x_multiply.py"
WORK_DIR="/home/member/Workspace/xuanphu/Work/Checkpoints/MS_RCNN_Multiply"
# WORK_DIR="/home/member/Workspace/xuanphu/Work/pretrained_models"

th=15
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
# CHECKPOINT_FILE="${WORK_DIR}/ms_rcnn_r50_caffe_fpn_1x.pth"

RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"
RESULT_FILE_JSON="${WORK_DIR}/epoch_${th}.json"
WRITE_TO="${WORK_DIR}/epoch_${th}.txt"

GPUS=2
CUDA_VISIBLE_DEVICES=2,3
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE} --json_out ${RESULT_FILE_JSON} --eval bbox segm
