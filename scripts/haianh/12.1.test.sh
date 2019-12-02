#!/usr/bin/env bash
CONFIG_FILE='ccdetection/configs/polarmask/polar_b1_bifpn.py'

WORK_DIR="work_dirs/polar_b1_bifpn"
th='latest'
CHECKPOINT_FILE="${WORK_DIR}/${th}.pth"
RESULT_FILE="${WORK_DIR}/${th}.pkl"
WRITE_TO="${WORK_DIR}/${th}.txt"

GPUS=8
CUDA_VISIBLE_DEVICES=0,1
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE} --eval bbox segm
# python mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}  --out ${RESULT_FILE} --show