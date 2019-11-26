CONFIG_FILE='ccdetection/configs/EfficientDet/retinamask_efficientdet.py'
GPUS=2
SEED=0

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    ./mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  

# export CUDA_VISIBLE_DEVICES=0
# python ./mmdetection/tools/train.py $CONFIG_FILE
# sh scripts/4.1_test.sh
# sh scripts/6.1_test.sh  