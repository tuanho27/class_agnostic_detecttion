CONFIG_FILE=configs/retina_mask/retinamask_r50_fpn_fp16.py
GPUS=2
SEED=0

# export CUDA_VISIBLE_DEVICES=2,3
# python -m torch.distributed.launch --nproc_per_node=$GPUS \
# 	--master_port=$((RANDOM + 10000)) \
#     ./tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  #--validate

# export CUDA_VISIBLE_DEVICES=0
python  ./tools/train.py $CONFIG_FILE

# sh scripts/4.1_test.sh