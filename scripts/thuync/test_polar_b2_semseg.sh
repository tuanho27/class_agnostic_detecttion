#!/usr/bin/env bash

CONFIG_FILE='ccdetection/configs/polarmask/polar_b2_semseg.py'
WORK_DIR='/home/member/Workspace/thuync/checkpoints/polar_b2_semseg/'

th=12
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"

GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
	--launcher pytorch --out ${RESULT_FILE} --eval bbox segm
