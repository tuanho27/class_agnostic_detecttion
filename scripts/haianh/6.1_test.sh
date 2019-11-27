#!/usr/bin/env bash
CONFIG_FILE='ccdetection/configs/fcos_mask/fcos_mask_r50_fp16.py'

WORK_DIR="work_dirs/fcos_mask_r50_fp16"
th=100
# CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
CHECKPOINT_FILE="${WORK_DIR}/latest.pth"
# CHECKPOINT_FILE='model_zoo/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_20190516-a36c0872.pth'
# CHECKPOINT_FILE='model_zoo/fcos_mstrain_640_800_r50_caffe_fpn_gn_2x_4gpu_20190516-f7329d80.pth'
RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"
WRITE_TO="${WORK_DIR}/epoch_${th}.txt"

GPUS=2
CUDA_VISIBLE_DEVICES=0,1
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	ccdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE} --eval bbox segm

# python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}  --out ${RESULT_FILE} --eval bbox segm

