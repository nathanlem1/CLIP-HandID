# CLIP-HandID

This code fine-tunes [CLIP](https://github.com/openai/CLIP) on hands datasets ([11k](https://sites.google.com/view/11khands) 
and [HD](http://www4.comp.polyu.edu.hk/~csajaykr/knuckleV2.htm)) for hand-based person identification. 
It also evaluates CLIP pre-trained models on the hands datasets in zero-shot fashion and then compare their performance with 
their fine-tuned counterparts. We fine-tune the image encoder only by keeping the text-encoder fixed or frozen. We used the 
vision transformer `ViT-B/16` and ResNet50 `RN50` CLIP backbone models for the experiments.


<!--
## Overview
In this paper, we propose a novel hand-based person recognition method for...

The proposed attention modules and the structure of CLIP-HandID are shown below.

![](./assets/MBA_Net.png)
-->


## Installation

Git clone this repo and install dependencies to have the same environment configuration as the one we used. Note that we trained 
the models on a single NVIDIA GeForce RTX 2080 Ti GPU.

```
git clone https://github.com/nathanlem1/CLIP-HandID.git.git
cd CLIP-HandID
pip install -r requirements.txt
```

You also need to install [CLIP](https://github.com/openai/CLIP) for CLIP zero-shot evaluation, particularly for running 
`eval_query_gallery_clip_zeroshot.py`. 

## Data Preparation
We use [11k](https://sites.google.com/view/11khands) and [HD](http://www4.comp.polyu.edu.hk/~csajaykr/knuckleV2.htm) datasets 
for our experiments.

1. To use the [11k](https://sites.google.com/view/11khands) dataset, you neet to create `11k` folder under the `CLIP-HandID` folder. Download dataset to `/CLIP-HandID/11k/` from [11k](https://sites.google.com/view/11khands) and extract it. You need to download both hand images and metadata (.csv file). The data structure will look like:

```
11k/
    Hands/
    HandInfo.csv
```
Then you can run following code to prepare the 11k dataset: 

```
python prepare_train_val_test_11k_r_l.py
```

2. To use the [HD](http://www4.comp.polyu.edu.hk/~csajaykr/knuckleV2.htm) dataset, you neet to create `HD` folder under the `CLIP-HandID` folder. Download dataset to `/CLIP-HandID/HD/` from [HD](http://www4.comp.polyu.edu.hk/~csajaykr/knuckleV2.htm) and extract it. You need to download the original images. The data structure will look like:

```
HD/
   Original Images/
   Segmented Images/
   ReadMe.txt
```
Then you can run following code to prepare the HD dataset: 
```
python prepare_train_val_test_hd.py
```

Read more [MBA-Net](https://ieeexplore.ieee.org/abstract/document/9956555) to get more information about the data split i.e. we followed similar data 
splitting fashion for this repo. 


## Train for fine-tuning
To train on the 11k dorsal right dataset, you need to run the following code on terminal:  

```
python train_finetune.py --data_dir ./11k/train_val_test_split_dorsal_r --f_name ./model_11k_d_r --data_type 11k --backbone_name ViT-B/16 --m_name clip_hand_vit
```

Please look into the `train_finetune.py` for more details. You need to provide the correct dataset i.e. right dorsal of 11k, left 
dorsal of 11k, right palmar of 11k, left palmar of 11k or HD dataset. You may need to change the name of `Original Images` in 
`HD/Original Images` to `Original_Images` so that it will look like `HD/Original_Images`. This helps to use it on command line 
to train the model on `HD` dataset. Thus, to train on the HD dataset, you need to run the following code on terminal:

```
python train_finetune.py --data_dir ./HD/Original_Images/train_val_test_split --f_name ./model_HD --data_type HD --backbone_name ViT-B/16 --m_name clip_hand_vit
```
You need to change vision transformer `ViT-B/16` to ResNet50 `RN50` CLIP backbone model to use ResNet50 based CLIP model for image encoder. 
You also need to change the output folder name `clip_hand_vit` to `clip_hand_rn50` when using `RN50` backbone CLIP image encoder model.  


## Evaluate

1. To evaluate using the CLIP pretrained model in zero-shot fashion, for instance, on the 11k dorsal right dataset, you need to run the following 
code on terminal:

```
python eval_query_gallery_clip_zeroshot.py --test_dir ./11k/train_val_test_split_dorsal_r --f_name ./model_11k_d_r --backbone_name ViT-B/16 
```
You need to change vision transformer `ViT-B/16` to ResNet50 `RN50` CLIP backbone model to use ResNet50 based CLIP model for image encoder. 
You also need to change the output folder name `clip_hand_vit` to `clip_hand_rn50` when using `RN50` backbone CLIP image encoder model.  


2. To evaluate using the finetuned model, for instance, on the 11k dorsal right dataset, you need to run the following code on terminal:

```
python eval_query_gallery_finetune.py --test_dir ./11k/train_val_test_split_dorsal_r --f_name ./model_11k_d_r --m_name clip_hand_vit
```

Please look into the `eval_query_gallery_finetune.py` for more details. In case you are using a command line, you can run on the HD dataset
after changing the name of `Original Images` in `HD/Original Images` to `Original_Images` so that it will look like `HD/Original_Images`, 
and then run the following code on terminal:

```
python eval_query_gallery_finetune.py --test_dir ./HD/Original_Images/train_val_test_split --f_name ./model_HD --m_name clip_hand_vit
```
You also need to change the output folder name `clip_hand_vit` to `clip_hand_rn50` when using `RN50` backbone CLIP image encoder model.

3. In addition, you can use `query_ranking_result_demo.py` to produce qualitative results.


<!---
## Citation

If you use this code for your research, please cite our paper.

```
@InProceedings{Nathanael_ICPR2022,
author = {Baisa, Nathanael L. and Williams, Bryan and Rahmani, Hossein and Angelov, Plamen and Black, Sue},
title = {Multi-Branch with Attention Network for Hand-Based Person Recognition},
booktitle = {The 26th International Conference on Pattern Recognition (ICPR)},
month = {Aug},
year = {2022}
}
```
-->
