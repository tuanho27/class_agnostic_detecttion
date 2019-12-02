CONFIG_FILE='ccdetection/configs/polarmask/polar_b2_semseg.py'
SEED=0
GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py ${CONFIG_FILE} --launcher pytorch --seed $SEED  #--validate
