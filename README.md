## SetUp
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
