# CONFIG_FILE='ccdetection/configs/rdsnet/rdsnet_b1_fpn_1x.py'
SEED=0
GPUS=1
export CUDA_VISIBLE_DEVICES=1

CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
# CONFIG_FILE='ccdetection/configs/faster_rcnn/fast_rcnn_r50_fpn_1x.py'
# python -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	--master_port=$((RANDOM + 10000)) \
#     mmdetection/tools/train.py ${CONFIG_FILE} --launcher pytorch --seed $SEED  #--validate
python mmdetection/tools/train.py ${CONFIG_FILE} 
