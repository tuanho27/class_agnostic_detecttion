## SetUp
```bash
conda create -n ccdet python=3.7 -y
conda activate ccdet

conda install pytorch=1.2 torchvision -c pytorch cython -y

#Setup mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

# Create symbolic link from ccdetection to mmdetection
cd ..
python ccdet_setup.py
conda install mpmath pandas -y
conda install -c conda-forge json_tricks -y
pip install torch_dct imagecorruptions albumentations pycocotools
pip install -e mmdetection/mmdet/models/backbones/pytorch-image-models
```