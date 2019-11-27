CONFIG_FILE='ccdetection/configs/EfficientDet/retinanet_efficient_idleblock.py'
GPUS=2
SEED=0

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    ./mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  

# python ./mmdetection/tools/train.py $CONFIG_FILE
# export CUDA_VISIBLE_DEVICES=0
# sh scripts/4.1_test.sh
# sh scripts/6.1_test.sh  