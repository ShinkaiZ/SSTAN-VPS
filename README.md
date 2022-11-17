# Semi-supervised Spatial Temporal Attention Network for Video Polyp Segmentation

## Introduction

This repository contains the **fully-supervised(fully-supervised for training, unsupervised for testing)** version in SUN-SEG dataset of:

Semi-supervised Spatial Temporal Attention Network for Video Polyp Segmentation, MICCAI 2022 [PDF](https://rdcu.be/cVRwv)

Although our work was based on semi-supervised learning, 
the fully-supervised version still achieves state-of-the-art results in most metrics.

## Usage

> This repository is based on [GewelsJI/VPS](https://github.com/GewelsJI/VPS),
> we strongly recommend you read their work first.

- Preparing the SUN-SEG dataset

  Please refer to [`DATA_PREPARATION`](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md)

- Prerequisites of environment:

  ```
    pip install torch yacs einops timm tqdm tensorboardX opencv-python albumentations
    ```

  Because we don't use the NS block, it is not necessary to compile it.

- Training:

    ```
    python ./scripts/my_train.py
    ```

- Testing:

  Downloading pre-trained weights and move it into `snapshot/SSTAN/epoch_15/SSTAN.pth`,
  which can be found in this download link: [MEGA](https://mega.nz/file/gHEAFLZb#ZkzpeTkUxjN8DQkssZyfIUe2gXK3DPxYDkuww7LrcS8)
    ```
    python ./scripts/my_test.py
    ```

- Evaluating:

   ```
    cd eval 
    bash eval.sh
    ```

## Results

You can directly download the prediction maps of our approach in this download link: [MEGA](https://mega.nz/folder/lPNXFTpa#YYCKjTZypauR8NyzhNh9gg)

- Quantitative comparison on two testing sub-datasets, SUN-SEG-Easy (Unseen) and SUN-SEG-Hard (Unseen):

Existing Methods: [source](https://github.com/GewelsJI/VPS/blob/main/assets/ModelPerformance.png)

<p align="center">
    <img src="./assets/ModelPerformance.png"/> <br />
</p>

Ours:

| Dataset              | Method            | Smeasure | meanEm | wFmeasure | meanFm | maxDice | meanSen |
|----------------------|-------------------|----------|--------|-----------|--------|---------|---------|
| SUN-SEG-Easy(Unseen) | 2022-MICCAI-SSTAN | 0.805    | 0.838  | 0.691     | 0.745  | 0.726   | 0.662   |
| SUN-SEG-Hard(Unseen) | 2022-MICCAI-SSTAN | 0.801    | 0.833  | 0.682     | 0.734  | 0.718   | 0.676   |


## Citations

If you feel this work is helpful, please cite our paper

    @inproceedings{zhao2022semi,
      title={Semi-supervised Spatial Temporal Attention Network for Video Polyp Segmentation},
      author={Zhao, Xinkai and Wu, Zhenhua and Tan, Shuangyi and Fan, De-Jun and Li, Zhen and Wan, Xiang and Li, Guanbin},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={456--466},
      year={2022},
      organization={Springer}
    }

## Changes

- vacs.py & vit_utils.py
  - add SSTAN
- dataloader.py & my_test.py
  - ensure that the frames input to the network are continuous
- my_train.py
  - change the loss function
  - fix the logging
- config.py

## Acknowledgements

- This codebase is based on [GewelsJI/VPS](https://github.com/GewelsJI/VPS). Thanks very much for their wonderful work!
