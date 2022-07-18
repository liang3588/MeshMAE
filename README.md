# meshMAE
This is the implementation of MeshMAE.
## Requirements

* python 3.9+
* CUDA 11.1+
* torch 1.11+

To install python requirements:
```setup
pip install -r requirements.txt
```


To install PyGem:
```
Please refer to Pygem (https://github.com/mathLab/PyGeM)
```

## Fetch Data
Here, we provide the download links of the datasets for pre-train, classification and segmentation. 

- ModelNet40 [here](https://cloud.tsinghua.edu.cn/f/af5c682587cc4f9da9b8/?dl=1)
- Humanbody [here](https://drive.google.com/file/d/1XaqMC8UrIZ_N77gN83PI3VK03G5IJskt/view?usp=sharing)
- COSEG-aliens [here](https://drive.google.com/file/d/12QCv2IUySoSzxeuvERGzgmE7YY3QzjfW/view?usp=sharing)
- ShapeNet [here](https://pan.quark.cn/s/eebb562558c6)



Please create a new folder 'datasets' in the main root, and put the downloaded datasets in this folder. And '--dataroot' in the 'xxx.sh' refers to the root of datasets. 

For example, the root of ModelNet40 should be:

```
--dataroot ./dataset/Manifold40-MAPS-96-3/ 
```


## Pretrain


* To pretrain on the ShapeNet dataset, you should run this command:
```
sh scripts/pretrain/train_pretrain_sn.sh
```
 


## Downstream Tasks

### Classification

* To train the classification model from scratch, you should run this command:

```
sh scripts/classification/train.sh
```

* To finetune the classification model, you should run this command:
```
sh scripts/classification/train_finetune.sh
```


### Segmentation

* To train the segmentation model from scratch, you should run this command:

```
sh scripts/segmentation/<...>/train.sh
```

* To finetune the segmentation model, you should run this command:

```
sh scripts/segmentation/<...>/train_finetune.sh
```


### Finetune note 
To finetune the model, you should create a folder 'checkpoints' in the main root, and put the
the pre-trained model in it. And '--checkpoint' in the 'train_finetune.sh' refer to the root the pre-trained model.

For example, the root of pre-trained model should be:

```
--checkpoint "./checkpoints/shapenet_pretrain.pkl"
```


## Models

Here, we provide the download links of the pre-trained models.

| Task              | Dataset        | Baseline                                                                                   | Finetune                                                                                   |      
|-------------------|----------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | [here](https://drive.google.com/file/d/1MOGlOfacoRL6ZrF4AAyB6akmio4Ek3es/view?usp=sharing) |---                                                                                        |
| Classification    | ModelNet40     | [here](https://drive.google.com/file/d/1gvqqnBR9EpWmoOgbe5lINc-6pfpim-uI/view?usp=sharing) | [here](https://drive.google.com/file/d/1kuo_Wz5lFDq7RZNUCI6LhK6q0szfyqfU/view?usp=sharing) |
| Segmentation      | HumanBody      | [here](https://drive.google.com/file/d/1WgPGiVqR891UF33S8s2QlsgWwyQLuilP/view?usp=sharing) | [here](https://drive.google.com/file/d/1q7yeBpMTuHhIeKXn8K_7ofAZ9pum9xot/view?usp=sharing)                                                                                   |
| Segmentation      | Coseg-alien    | [here](https://drive.google.com/file/d/1UyDwkDtkU9eFAuv8nPT_M35Y6SnalVTI/view?usp=sharing) | [here](https://drive.google.com/file/d/1D9tvEwjSb2lEc4acGAtmgJEGttP2SIzC/view?usp=sharing) |
