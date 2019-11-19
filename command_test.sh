#!/usr/bin/env bash

CONFIG_FILE="ccdetection/configs/retina_mask/retinamask_r50_fpn_1x.py"
WORK_DIR="/home/user/thuync/checkpoints/retinamask_r50_fpn_1x"

th=6
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"
WRITE_TO="${WORK_DIR}/epoch_${th}.txt"

GPUS=2
CUDA_VISIBLE_DEVICES=0,1
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE} --eval bbox
# $PYTHON mmdetection/tools/voc_eval.py ${RESULT_FILE} ${CONFIG_FILE} --iou-th=0.5 > $WRITE_TO
# cat $WRITE_TO
