## SetUp
```bash
conda create -n ccdetpose python=3.7 -y
conda activate ccdetpose

conda install pytorch=1.2 torchvision -c pytorch -y
conda install cython pyyaml -y

#Setup mmdetection
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard 4d84161f142b7500089b0db001962bbc07aa869d
pip install -v -e .

# Create symbolic link from ccdetection to mmdetection
cd ..
python ccdet_setup.py
conda install mpmath pandas -y
conda install -c conda-forge json_tricks -y
pip install torch_dct imagecorruptions albumentations pycocotools
pip install -e mmdetection/mmdet/models/backbones/pytorch-image-models
```
