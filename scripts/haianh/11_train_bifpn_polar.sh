CONFIG_FILE='ccdetection/configs/polarmask/polar_b1_bifpn.py'
SEED=0
GPUS=2
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py ${CONFIG_FILE} --launcher pytorch --seed $SEED  #--validate
# python mmdetection/tools/train.py ${CONFIG_FILE}