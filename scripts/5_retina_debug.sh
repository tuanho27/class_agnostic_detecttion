CONFIG_FILE=configs/retinanet_r50_fpn_1x.py
GPUS=4
SEED=0

export CUDA_VISIBLE_DEVICES=0

python  ./tools/train.py $CONFIG_FILE

