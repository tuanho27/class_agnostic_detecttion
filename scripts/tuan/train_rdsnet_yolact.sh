# CONFIG_FILE='ccdetection/configs/rdsnet/rdsnet_b1_fpn_1x.py'
SEED=0
GPUS=2
export CUDA_VISIBLE_DEVICES=2,3

# CONFIG_FILE='ccdetection/configs/retina_mask/retinamask_b1_fpn_1x.py'
CONFIG_FILE='ccdetection/configs/fcos_mask/fcos_mask_b1_fp16.py'

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py ${CONFIG_FILE} --launcher pytorch --seed $SEED  #--validate
