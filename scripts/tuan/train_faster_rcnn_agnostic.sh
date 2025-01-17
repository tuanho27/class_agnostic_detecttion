SEED=0
GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco14.py'
# CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py ${CONFIG_FILE} --launcher pytorch --seed $SEED  #--validate

# CUDA_VISIBLE_DEVICES=0 python mmdetection/tools/train.py ${CONFIG_FILE} 
