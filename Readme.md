# FYP-Medical-Image-Segmentation-using-U-Net-Architecture
Dive into the future of medical image analysis with our GitHub repo! Discover the U-Net++ architecture's potential for precise and efficient FYP-Medical Image Segmentation
Welcome to the **FYP-Medical Image Segmentation using U-Net++ Architecture** repository!

## About

This repository contains the code and resources for our final year project (FYP) focusing on advanced medical image segmentation. We explore the potential of the U-Net++ architecture to achieve precise and efficient segmentation of complex medical images.
This repository contains code for an image segmentation model based on **UNet++**: [A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch.

[**NEW**] Added **_BCEDiceLoss, LovaszHingeLoss, IoULoss, DiceLoss and their CombinedLoss_**.

[**NEW**] Added evaluation metrics like **_iou_score, dice_coef, accuracy, precision, recall, and f1_score_**.

[**NEW**] Added preprocessing files of Datasets like [CVC-612](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) and [DRIVE (Digital Retinal Images for Vessel Extraction)](https://paperswithcode.com/dataset/drive)

## Key Features

- **U-Net++ Architecture**: We leverage the power of U-Net and U-Net++ for improved segmentation accuracy.
- **Cutting-Edge Techniques**: Incorporating state-of-the-art deep learning methods for medical image analysis.
- **Collaboration**: Join us in enhancing medical diagnostics and treatment through innovative segmentation solutions.

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.10 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```
## Training on datasets
1. Download Data Science Bowl 2018 dataset from [here](https://www.kaggle.com/c/data-science-bowl-2018/data) to inputs/ and unzip. The file structure is the following:
```
inputs
└── data-science-bowl-2018
    ├── stage1_train
    |   ├── 00ae65...
    │   │   ├── images
    │   │   │   └── 00ae65...
    │   │   └── masks
    │   │       └── 00ae65...            
    │   ├── ...
    |
    ...
```
1.1 Download Polyp dataset from [here](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) to inputs/ and unzip. The file structure is the following:
```
inputs
└── images
    ├── 1.png
└── masks
    ├── 1.png
```

1.2 Download Retina dataset from [here](https://paperswithcode.com/dataset/drive) to inputs/ and unzip. The file structure is the following:
```
inputs
└── images
    ├── 27_training.tif
└── masks
    ├── 28_manual1.gif
```
2. Preprocess.
```sh
# for Data Science Bowl 2018
python preprocess_dsb2018.py
# for Retina Dataset
python preprocess_retina.py
# for Polyp Dataset
python preprocess_polyp.py
```
3. Train the model.
```sh
python train.py --dataset dsb2018_96 --arch NestedUNet
```
4. Evaluate.
```sh
python val.py --name dsb2018_96_NestedUNet_woDS
```
5. Plot Graphs.
```sh
python plot.py
```
### (Optional) Using CombinedLoss
1. Train the model with CombinedLoss.
```
python train.py --dataset dsb2018_96 --arch NestedUNet --loss CombinedLoss
```
## Training on original dataset
Make sure to put the files as the following structure (e.g. the number of classes is 2):
```
inputs
└── <dataset name>
    ├── images
    |   ├── 0a7e06.jpg
    │   ├── 0aab0a.jpg
    │   ├── 0b1761.jpg
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 0a7e06.png
        |   ├── 0aab0a.png
        |   ├── 0b1761.png
        |   ├── ...
        |
        └── 1
            ├── 0a7e06.png
            ├── 0aab0a.png
            ├── 0b1761.png
            ├── ...
```

1. Train the model.
```
python train.py --dataset <dataset name> --arch NestedUNet --img_ext .jpg --mask_ext .png
```
2. Evaluate.
```
python val.py --name <dataset name>_NestedUNet_woDS
```
## Results
### DSB2018 (256x256)

Here is the results on DSB2018 dataset (256x256) with CombinedLoss.

| Model                           |   IoU   |  Dice   | Accuracy| Precision   |  Recall   |  F1 Score   |
|:------------------------------- |:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Nested U-Net                    |  0.8407 | 0.8739  | 0.9755  | 0.8490  | 0.9344  | 0.9132  |
## License
This project is licensed under the [MIT License](LICENSE).
