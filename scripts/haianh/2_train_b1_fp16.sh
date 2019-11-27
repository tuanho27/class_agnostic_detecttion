CONFIG_FILE=./configs/fp16/retinamask_b1_fpn_1x.py
GPUS=4
SEED=0

# export CUDA_VISIBLE_DEVICES=0,1
# python -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	--master_port=$((RANDOM + 10000)) \
#     ./tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  #--validate

# export CUDA_VISIBLE_DEVICES=0
python  ./tools/train.py $CONFIG_FILE
