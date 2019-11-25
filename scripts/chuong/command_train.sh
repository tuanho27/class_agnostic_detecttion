
CONFIG_FILE='ccdetection/configs/EfficientDet/retinamask_efficientdet.py'
WORK_DIR="/home/chuong/Workspace/Experiments/Retina_EffDet"
DATA_ROOT="/home/chuong/Workspace/dataset/coco/"
# WORK_DIR="/home/member/Workspace/chuong/Experiments/FARetina_r50_fpn"
# DATA_ROOT="/home/member/Workspace/dataset/coco/"

GPUS=2
SEED=0

# export CUDA_VISIBLE_DEVICES=2,3

# python -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	--master_port=$((RANDOM + 10000)) \
#     mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch \
#     --work_dir ${WORK_DIR} --data_root ${DATA_ROOT} --seed $SEED  #--validate

export CUDA_VISIBLE_DEVICES=1
python  mmdetection/tools/train.py $CONFIG_FILE --work_dir ${WORK_DIR} --data_root ${DATA_ROOT}
