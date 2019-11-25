CONFIG_FILE='ccdetection/configs/ms_rcnn_r50_caffe_fpn_1x_multiply.py'
# RESUME_FILE='/home/member/Workspace/xuanphu/Work/pretrained_models/epoch_5.pth'
SEED=0
GPUS=2
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=$GPUS \
	--master_port=$((RANDOM + 10000)) \
    mmdetection/tools/train.py $CONFIG_FILE --launcher pytorch --seed $SEED  #--validate
