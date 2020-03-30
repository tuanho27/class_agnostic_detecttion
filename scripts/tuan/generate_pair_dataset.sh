SEED=0
GPUS=1
export CUDA_VISIBLE_DEVICES=0 
num_sample=-1 # set -1 for all datasets

# CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
# txt_file='./list_pairs_img_voc2007.txt' 
# txt_eval_file='./list_pairs_img_test_voc2007.txt' 

CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco14.py'
txt_file='./list_pairs_img_coco2014.txt' 
txt_eval_file='./list_pairs_img_test_coco2014.txt'  

rm -f $txt_file $txt_eval_file

## for training data 
# python mmdetection/tools/generate_pair_dataset.py ${CONFIG_FILE} --output $txt_file --num_sample=$num_sample
## for validation data
python mmdetection/tools/generate_pair_dataset.py ${CONFIG_FILE} --output $txt_eval_file --validate --num_sample=$num_sample


