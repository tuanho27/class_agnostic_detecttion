## SetUp Env
```bash
git clone git@gitlab.com:chuong98vt/ccdetpose.git
cd ccdetpose

conda create -n ccdetpose python=3.7 -y
conda activate ccdetpose

conda install cython pyyaml -y
conda install mpmath pandas tqdm -y
conda install -c conda-forge json_tricks -y
pip install pytest torch_dct imagecorruptions albumentations pycocotools
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch -y
pip install git+https://github.com/vnbot2/pyson xxhash

#Setup mmdetection
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard 4d84161f142b7500089b0db001962bbc07aa869d
python setup.py develop

# Create symbolic link from ccdetection to mmdetection
cd ..
python ccsetup.py
pip install -e mmdetection/mmdet/models/backbones/pytorch-image-models
```

## Train & Test
0. To generate dataset
COCO:
- Uncomment line 10 in file: ccdetection/mmdet/datasets/coco_pair.py, and comment line 11 
- Uncomment line 553 - 569 in file:  ccdetection/mmdet/datasets/custom.py
VOC:
- Uncomment line 92 in file: ccdetection/mmdet/datasets/xml_style.py, and comment line 91
- Uncomment line 553 - 569 in file:  ccdetection/mmdet/datasets/custom.py

Then change the config file in ./scripts/tuan/generate_pair_dataset.sh and execute this file, wait until enough expected sample

1. To train dataset
- Change setting in ./scripts/tuan/train_faster_rcnn_agnostic.sh and then execute

2. To test inference pair of images
- Change setting in ./scripts/tuan/test_faster_rcnn_agnostic.sh and then execute
 