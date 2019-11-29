CONFIG_FILE='ccdetection/configs/polarmask/polar_b1.py'
GPUS=2
SEED=0
python ./mmdetection/tools/train.py $CONFIG_FILE
export CUDA_VISIBLE_DEVICES=0,1
# python -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	--master_port=$((RANDOM + 10000)) \
#     ./mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  

