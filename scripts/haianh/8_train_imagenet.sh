cd ccdetection/mmdet/models/backbones/pytorch-image-models
model=efficientnet_b1_idle
./distributed_train.sh 4 /computer/member/Workspace/dataset/ImageNet/ --model $model  -b 128 --sched step --epochs 500 --decay-epochs 3 --decay-rate 0.963 --opt rmsproptf --opt-eps .001 -j 8 --warmup-epochs 5 --weight-decay 1e-5 --drop 0.2 --color-jitter .06 --model-ema --lr .128 --img-size 320
# python validate.py /computer/member/Workspace/dataset/ImageNet --model model --checkpoint output/train/20191127-100922-efficientnet_b2_idle-260/checkpoint-285.pth.tar