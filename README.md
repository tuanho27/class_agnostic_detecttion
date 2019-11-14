## SetUp
```bash
conda create -n ccdet python=3.7 -y
conda activate ccdet

conda install pytorch=1.2 torchvision -c pytorch -y
conda install cython -y

#Setup mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

# Create symbolic link from ccdetection to mmdetection
cd ..
python setup.py
conda install mpmath pandas -y
pip install torch_dct
pip install -e mmdetection/mmdet/models/backbones/pytorch-image-models
```