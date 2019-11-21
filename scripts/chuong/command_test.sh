#!/usr/bin/env bash

CONFIG_FILE="ccdetection/configs/mask_rcnn_r50_fpn_1x.py"
WORK_DIR="/home/chuong/Workspace/Experiments/mask_rcnn_r50_fpn"
DATA_ROOT="/home/chuong/Workspace/dataset/coco/"
th=12
CHECKPOINT_FILE="${WORK_DIR}/epoch_${th}.pth"
RESULT_FILE="${WORK_DIR}/epoch_${th}.pkl"
WRITE_TO="${WORK_DIR}/epoch_${th}.txt"


GPUS=2
CUDA_VISIBLE_DEVICES=0,1
PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
	mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch \
		--out ${RESULT_FILE} --work_dir ${WORK_DIR} --data_root ${DATA_ROOT} --eval bbox segm \
		 >> ${WRITE_TO}

# python mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
# 		--out ${RESULT_FILE} --work_dir ${WORK_DIR} --data_root ${DATA_ROOT} --eval bbox segm \
# 		>> ${WRITE_TO}