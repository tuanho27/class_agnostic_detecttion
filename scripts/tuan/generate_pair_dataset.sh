SEED=0
GPUS=1
export CUDA_VISIBLE_DEVICES=2 #,2,3

# CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco14.py'

python mmdetection/tools/generate_pair_dataset.py ${CONFIG_FILE} 
