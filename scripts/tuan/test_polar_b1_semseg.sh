#!/usr/bin/env bash

#CONFIG_FILE='ccdetection/configs/polarmask/polar_b1_semseg.py'
#CONFIG_FILE='ccdetection/configs/rdsnet/rdsnet_b1_fpn_1x.py'
#WORK_DIR='./work_dirs/rdsnet_b1_fpn_1x/'
#CONFIG_FILE='ccdetection/configs/retina_mask/retinamask_b1_fpn_1x.py'
CONFIG_FILE='ccdetection/configs/fcos_mask/fcos_mask_b1_fp16.py'
WORK_DIR='work_dirs/fcos_mask_b1_fp16_seg'

th=12
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"

GPUS=2
export CUDA_VISIBLE_DEVICES=2,3
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
	--launcher pytorch --out ${RESULT_FILE} --eval bbox segm
