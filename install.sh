rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard 4d84161f142b7500089b0db001962bbc07aa869d
pip install -e .
cd ..
rm -rf ~/.origin_mmdetection && cp -r mmdetection ~/.origin_mmdetection

# Create symbolic link from ccdetection to mmdetection
python ccsetup.py
conda install mpmath pandas -y
conda install -c conda-forge json_tricks -y
pip install torch_dct imagecorruptions albumentations pycocotools xxhash
pip install -e mmdetection/mmdet/models/backbones/pytorch-image-models
pip install git+https://github.com/vnbot2/pyson
