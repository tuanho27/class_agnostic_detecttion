CONFIG_FILE='ccdetection/configs/fcos_mask/fcos_mask_r50_fp16.py'
GPUS=2
SEED=0


python ./mmdetection/tools/train.py $CONFIG_FILE

