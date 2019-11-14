# CONFIG_FILE="ccdetection/configs/clouds/htc_hrnetv2p_w32_cloud.py"
# CONFIG_FILE="ccdetection/configs/clouds/htc_x50gcb_fpn_cloud.py"
# CONFIG_FILE="ccdetection/configs/clouds/htc_x50gcb_nasfpn_cloud.py"
# CONFIG_FILE="ccdetection/configs/clouds/reppoints_moment_x50_dcn_nasfpn_cloud.py"
CONFIG_FILE="ccdetection/configs/clouds/retina_x50gcb_nasfpn_cloud.py"
GPUS=2
SEED=0

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  #--validate

# export CUDA_VISIBLE_DEVICES=1
# python  mmdetection/tools/train.py $CONFIG_FILE
