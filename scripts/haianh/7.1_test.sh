#!/usr/bin/env bash
CONFIG_FILE='ccdetection/configs/EfficientDet/retinanet_efficient_idleblock.py'

WORK_DIR="work_dirs/retinanet_efficient_idleblock"
th='latest'

CHECKPOINT_FILE="${WORK_DIR}/${th}.pth"
RESULT_FILE="${WORK_DIR}/${th}.pkl"
WRITE_TO="${WORK_DIR}/${th}.txt"

GPUS=2
CUDA_VISIBLE_DEVICES=0,1
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE} --eval bbox segm
# python mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}  --out ${RESULT_FILE} --show