CONFIG_FILE='ccdetection/configs/EfficientDet/retinamask_efficientdet.py'
GPUS=1
SEED=0

# python -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	--master_port=$((RANDOM + 10000)) \
#     ./mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  

python ./mmdetection/tools/train.py $CONFIG_FILE
# export CUDA_VISIBLE_DEVICES=0
# sh scripts/4.1_test.sh
sh scripts/haianh/7.1_test.sh  