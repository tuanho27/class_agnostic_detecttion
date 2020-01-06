CONFIG_FILE='ccdetection/configs/polarmask/polar_b1_semseg.py'
SEED=0
GPUS=2
export CUDA_VISIBLE_DEVICES=0,1 #,1,2,3

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py ${CONFIG_FILE} --launcher pytorch --seed $SEED  #--validate
