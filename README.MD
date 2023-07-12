# A Robust Feature Downsampling Module for Remote Sensing Visual Tasks [TGRS 2023]

This is the official Pytorch/Pytorch implementation of the paper: <br/>
> [**A Robust Feature Downsampling Module for Remote Sensing Visual Tasks**](https://ieeexplore.ieee.org/document/10142024)      
> Wei Lu; Si-Bao Chen; Jin Tang; Chris H. Q. Ding; Bin Luo        
> *IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2023*
> 

--- 
We propose a new and universal downsampling module named **robust feature downsampling (RFD)**.

<p align="center">
<img src="https://github.com/lwCVer/RFD/blob/main/figures/fig3_all_model_1.pdf" width=100% 
class="center">
</p>

## Image Classification
### 1. Dependency Setup
Create a new conda virtual environment
```
conda create -n RFD python=3.7 -y
conda activate RFD
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
Clone this repo and install the required packages:
```
git clone https://github.com/lwCVer/RFD
cd RFD/
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the [NWPU-RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45) classification dataset and structure the data as follows:
```
/path/to/NWPU-RESISC45/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```




### 3. Pre-trained Models on NWPU-RESISC45 (initial / +RFD) 

| name (initial / +RFD) |    type     |  #params (M)  |   FLOPs (G)   | Throughput (fps) |     Top-1 acc     |                                                                                       model                                                                                        |
|:---------------------:|:-----------:|:-------------:|:-------------:|:----------------:|:-----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     GFNet-H Tiny      |     FFT     | 14.60 / 15.68 |  2.05 / 2.43  |  2693 /   2128   |   92.27 / 94.76   |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/GFNet-H_tiny.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/GFNet-H_RFD_tiny.pth)   |
|     GFNet-H Small     |     FFT     | 31.43 / 33.05 |  4.59 / 5.34  |   2405 /  2466   |   93.40 / 95.11   |  [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/GFNet-H_small.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/GFNet-H_RFD_small.pth)  |
|     GFNet-H Base      |     FFT     | 53.01 / 55.43 |  8.53 / 9.28  |   2098 / 1886    | 94.17 / **95.46** |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/GFNet-H_base.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/GFNet-H_RFD_base.pth)   |
|      AS-MLP Tiny      |     MLP     | 27.55 / 29.96 |  4.39 / 5.14  |   1505 / 1571    | 95.37 / **96.05** |    [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/AS-MLP_tiny.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/AS-MLP_RFD_tiny.pth)    |
|     AS-MLP Small      |     MLP     | 48.86 / 51.27 |  8.57 / 9.32  |   1073 / 1019    |   95.27 / 96.00   |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/AS-MLP_small.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/AS-MLP_RFD_small.pth)   |
|      AS-MLP Base      |     MLP     | 86.77 / 91.05 | 15.2 / 16.44  |    861 / 830     |   95.63 / 95.94   |    [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/AS-MLP_base.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/AS-MLP_RFD_base.pth)    |
|       Swin Tiny       | Transformer | 27.55 / 29.97 |  4.36 / 5.11  |   2469 / 2313    |   93.52 / 96.10   |      [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_tiny.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_RFD_tiny.pth)      |
|      Swin Small       | Transformer | 48.87 / 51.28 |  8.52 / 9.27  |   1995 / 1762    |   93.37 / 96.16   |     [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_small.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_RFD_small.pth)     |
|       Swin Base       | Transformer | 86.79 / 91.06 | 15.14 / 16.37 |   1975 / 1734    | 93.19 / **96.24** |      [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_base.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_RFD_base.pth)      |
|      CSWin Tiny       | Transformer | 21.83 / 22.05 |  4.08 / 4.35  |   1303 / 1127    |   93.05 / 96.11   |     [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/CSWin_tiny.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/CSWin_RFD_tiny.pth)     |
|      CSWin Small      | Transformer | 34.15 / 34.37 |  6.40 / 6.67  |    682 / 579     |   93.56 / 96.11   |    [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/CSWin_small.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/CSWin_RFD_small.pth)    |
|      CSWin Base       | Transformer | 76.65 / 77.12 | 14.36 / 14.87 |    481 / 406     | 94.49 / **96.29** |     [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/CSWin_base.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/CSWin_RFD_base.pth)     |
|     Swin V2 Tiny      | Transformer | 27.61 / 30.03 |  3.33 / 4.08  |   2009 / 1987    |   94.65 / 96.46   |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_V2_tiny.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_V2_RFD_tiny.pth)   |
|     Swin V2 Small     | Transformer | 48.99 / 51.41 |  6.47 / 7.22  |    1273 / 827    | 95.22 / **96.84** |  [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_V2_small.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_V2_RFD_small.pth)  |
|     Swin V2 Base      | Transformer | 86.94 / 91.22 | 11.48 / 12.71 |    1104 / 682    |   95.63 / 96.61   |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_V2_base.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Swin_V2_RFD_base.pth)   |
|    Mixformer Tiny     |   Hybrid    |  5.10 / 5.25  |  0.39 / 0.44  |   1287 / 1192    |   94.87 / 95.30   |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Mixformer_B0.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Mixformer_RFD_B0.pth)   |
|    Mixformer Small    |   Hybrid    | 9.89 / 10.17  |  0.88 / 0.95  |    1018 / 975    |   95.41 / 96.03   |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Mixformer_B2.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Mixformer_RFD_B2.pth)   |
|    Mixformer Base     |   Hybrid    | 34.80 / 35.85 |  3.44 / 3.56  |    830 / 722     | 95.76 / **96.37** |   [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/Mixformer_B4.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/Mixformer_RFD_B4.pth)   |
|     ConvNeXt Tiny     |     CNN     | 28.85 / 30.27 |  4.47 / 5.22  |   3109 / 2566    |   93.70 / 95.48   |  [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/ConvNeXt_tiny.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/ConvNeXt_RFD_tiny.pth)  |
|    ConvNeXt Small     |     CNN     | 49.49 / 51.90 |  8.70 / 9.45  |   2734 / 2478    |   93.90 / 95.48   | [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/ConvNeXt_small.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/ConvNeXt_RFD_small.pth) |
|     ConvNeXt Base     |     CNN     | 87.61 / 91.89 | 15.38 / 16.61 |   2726 / 2520    | 95.02 / **96.13** |  [initial](https://github.com/lwCVer/RFD/releases/download/pre-train/ConvNeXt_base.pth) / [+RFD](https://github.com/lwCVer/RFD/releases/download/pre-train/ConvNeXt_RFD_base.pth)  |


### 4. Training

Swin V2 Tiny training on RESISC45:
```
python train.py 
```
To train other models, `train.py` need to be changed.        
  

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [mmdetection](https://github.com/open-mmlab/mmdetection) repositories.

## Citation
If you find this repository helpful, please consider citing:
```
@article{lu2023robust,
  title={A Robust Feature Downsampling Module for Remote Sensing Visual Tasks},
  author={Lu, Wei and Chen, Si-Bao and Tang, Jin and Ding, Chris HQ and Luo, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--12},
  year={2023},
  publisher={IEEE}
}
```
