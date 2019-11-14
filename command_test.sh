#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG_FILE=ccdetection/configs/clouds/reppoints_moment_x50_dcn_nasfpn_cloud.py
WORK_DIR=/home/chuong//Workspace/Experiments/cloud/reppoints_moment_x50_dcn_nasfpn_cloud/v1

th=12
CHECKPOINT_FILE=${WORK_DIR}/epoch_${th}.pth
RESULT_FILE=${WORK_DIR}/epoch_${th}_025.pkl
WRITE_TO=${WORK_DIR}/epoch_${th}_025.txt

GPUS=2
CUDA_VISIBLE_DEVICES=0,1

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
        mmdetection/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --launcher pytorch --out ${RESULT_FILE}
$PYTHON mmdetection/tools/voc_eval.py ${RESULT_FILE} ${CONFIG_FILE} --iou-th=0.5 > $WRITE_TO
cat $WRITE_TO