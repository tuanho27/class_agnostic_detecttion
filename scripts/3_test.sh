#!/usr/bin/env bash

CONFIG_FILE="configs/fp16/retinamask_b1_fpn_1x.py"
WORK_DIR="work_dirs/retinamask_b1_fpn_1x"

th=99
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"
WRITE_TO="${WORK_DIR}/epoch_${th}.txt"

GPUS=2
CUDA_VISIBLE_DEVICES=0,1
PYTHON=${PYTHON:-"python"}

# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE} --eval bbox segm
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}  --out ${RESULT_FILE} --eval bbox segm
