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
1. First, we need to create an annotation for dataset of pair images. 
    1.1. We loop over the annotations in COCO and VOC datasets, to collect the image ID of each class.

    COCO:
    - Uncomment line 10 in file: ccdetection/mmdet/datasets/coco_pair.py, and comment line 11 

    VOC:
    - Uncomment line 91 in file: ccdetection/mmdet/datasets/xml_style.py, and comment line 91

    Then ajust the config path for each dataset and execute as follow:
    ```bash
    num_sample=-1 # set -1 for all datasets

    CONFIG_FILE='ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
    txt_file='./list_pairs_img_voc2007.txt' 
    txt_eval_file='./list_pairs_img_test_voc2007.txt' 

    ## for create training data 
    python mmdetection/tools/generate_pair_dataset.py ${CONFIG_FILE} --output $txt_file --num_sample=$num_sample
    ## for create validation data
    #python mmdetection/tools/generate_pair_dataset.py ${CONFIG_FILE} --output $txt_eval_file --validate --num_sample=$num_sample
    ```
    And execute below command
    ```bash 
    ./scripts/tuan/generate_pair_dataset.sh 
    ```

    After finished the generation process, the script will create two text file as outputs each types of datasets, which contain pair image ids, namely:

    VOC:
    + ./list_pairs_img_voc2007.txt : train dataset
    + ./list_pairs_img_test_voc2007.txt: validation dataset

    COCO:
    + ./list_pairs_img_coco2014.txt : train dataset
    + ./list_pairs_img_test_coco2014.txt: validation dataset

    This dataset class is inherited from CoCoDataset or VOCDataset Class except it has:
    + `ignore_classes`: list of class IDs that we don't want to make a pair. This is mainly for debuging or for hold out a subset classes to test unknown objects.


2. To train the model, run the one of the following commands:
    ```bash
      ./scripts/tuan/train_faster_rcnn_agnostic.sh
    ```
    Corespondingly, the model configs are:
    + `ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco14.py`, using the detector `TwoStagePairDetector`. This is Faster-RCNN + Co-Detection (train Region Proposal + BBox + Codet) for COCO datasets
    + `ccdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py`,  using the detector `TwoStagePairDetector`. This is Faster-RCNN + Co-Detection (train Region Proposal + BBox + Codet) for VOC datasetss

    Following is the list of important options:
    + `dataset_type`: VOCPairDataset or CocoPairDataset
    + `debug=False`: There are two modes: Debug (just train over fit a couple of images) and not_Debug(train whole dataset).
    + `rpn_head`: is the same as in the convetional FasterRCNN and MaskRCNN. 
    + `mask_roi_extractor` and `mask_head`: are the same as in the convetional FasterRCNN and MaskRCNN.
    + `matching_head` is the only new module, which apply several convs and linear layers to the ROI extracted from `codet_roi_extractor` to extract an embedded vector, including `SiameseMatchingHead` and `RelationMatchingHead`
      - `num_convs`: number of conv layers applied to ROIs.
      - `in_channels` number of features of ROIs.
      - `feat_channels`: number of feature output after applying convs.
    + `train_cfg` is the same as in the convetional FasterRCNN. 
    + `test_cfg` is the new config with following paras:
      - `nms_pre`: non maxima suppression config for each image.
      - `nms_post`: non maxima suppression config for each image.
      - `max_num`: Maximum number of proposal selected from each image.
      - `nms_thr`: iou_thesh hold for non maxima suppression 
      - `topk_pair_select`: Maximumn number of pairs. 
      - `mode`: mode for evaluation or inference. 
      - `matching_head`: choose matching head use for inference. 
      - `score_thr`: Final scores of related pair features to select. 

3. To test the model, run the one of the following commands:

    Please change the option and config files in the bash file 
    ```bash
      ./scripts/tuan/test_faster_rcnn_agnostic.sh
    ```
    Output images will be saved as ./outputs folder
    You can select the mode inference for visualization, and evaluation to measure Recall & Precision Values
    