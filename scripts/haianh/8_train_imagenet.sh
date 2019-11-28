cd ccdetection/mmdet/models/backbones/pytorch-image-models

# ./distributed_train.sh 4 /computer/member/Workspace/dataset/ImageNet --model efficienet_b2_ --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 -j 4
./distributed_train.sh 4 /computer/member/Workspace/dataset/ImageNet --model efficientnet_b2_idle --sched cosine --epochs 300 --warmup-epochs 5 --lr 0.2 --reprob 0.5 --remode pixel --batch-size 128 -j 4


python validate.py /computer/member/Workspace/dataset/ImageNet --model efficientnet_b2_idle --checkpoint output/train/20191127-100922-efficientnet_b2_idle-260/checkpoint-285.pth.tar