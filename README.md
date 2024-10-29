# Regularizing with Pseudo-Negatives for Continual Self-Supervised Learning [ICML 2024]

Sungmin Cha, Kyunghyun Cho and Taesup Moon

Paper: [[Link](https://proceedings.mlr.press/v235/cha24a.html)]

Abstract: We introduce a novel Pseudo-Negative Regularization (PNR) framework for effective continual self-supervised learning (CSSL). Our PNR leverages pseudo-negatives obtained through model-based augmentation in a way that newly learned representations may not contradict what has been learned in the past. Specifically, for the InfoNCE-based contrastive learning methods, we define symmetric pseudo-negatives obtained from current and previous models and use them in both main and regularization loss terms. Furthermore, we extend this idea to non-contrastive learning methods which do not inherently rely on negatives. For these methods, a pseudo-negative is defined as the output from the previous model for a differently augmented version of the anchor sample and is asymmetrically applied to the regularization term. Extensive experimental results demonstrate that our PNR framework achieves state-of-the-art performance in representation learning during CSSL by effectively balancing the trade-off between plasticity and stability.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sy-con-symmetric-contrastive-loss-for/continual-self-supervised-learning-on-cifar-4)](https://paperswithcode.com/sota/continual-self-supervised-learning-on-cifar-4?p=sy-con-symmetric-contrastive-loss-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sy-con-symmetric-contrastive-loss-for/continual-self-supervised-learning-on-cifar-3)](https://paperswithcode.com/sota/continual-self-supervised-learning-on-cifar-3?p=sy-con-symmetric-contrastive-loss-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sy-con-symmetric-contrastive-loss-for/continual-self-supervised-learning-on-2)](https://paperswithcode.com/sota/continual-self-supervised-learning-on-2?p=sy-con-symmetric-contrastive-loss-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sy-con-symmetric-contrastive-loss-for/continual-self-supervised-learning-on-1)](https://paperswithcode.com/sota/continual-self-supervised-learning-on-1?p=sy-con-symmetric-contrastive-loss-for)

-------


## Environment

We followed the experimental environment of [CaSSLe](https://github.com/DonkeyShot21/cassle) as closely as possible. For details, please refer 'cassle.yml'.

-------

## Implementation Guide

### Setting Up the Environment and Folders
1. Create the environment: conda env create -f cassle.yml
2. Clone the repository: git clone https://github.com/csm9493/PNR.git
3. Navigate to the project directory: cd PNR
4. Create folders: mkdir experiments trained_models wandb
5. Run [Wandb](https://docs.wandb.ai/quickstart/?_gl=1*1ti67r4*_ga*NjA0NjAxMDY5LjE3Mjk1MjgxMzc.*_ga_JH1SJHJQXJ*MTczMDE2OTg5MS4zLjAuMTczMDE2OTg5NS41Ni4wLjA.*_ga_GMYDGNGKDT*MTczMDE2OTg5MS40LjAuMTczMDE2OTg5MS4wLjAuMA..*_gcl_au*MTEzNjMxODEyOC4xNzI5MjgxODkx).

### CSSL Experiments using CIFAR-100
1. Navigate to the project directory: cd PNR
2. Check 'run_cifar100.sh' and change `PROJECT`, `ENTITY` and `DATA_DIR`.
3. Run the script: ./run_cifar100.sh

> Note that each `.sh` file for experiments using the **ImageNet-100** and **DomainNet** datasets contains code for two stages: "**Step 1: CSSL**" and "**Step 2: Linear Evaluation**." After completing Step 1 (CSSL), remember to update the model path in `PRETRAINED_PATH` before starting the linear evaluation.
> 
### CSSL Experiments using ImageNet-100 (class- and data-IL)
1. Navigate to the project directory: cd PNR
2. Download [ImageNet-100 dataset](https://www.kaggle.com/datasets/ambityga/imagenet100) and place it in `YOUR_DATASET_PATH`.
- For **Class-IL** experiments
3. Check run_imagenet100_class.sh and change `PROJECT`, `ENTITY`, `DATA_DIR`, `YOUR_IMAGENET100_TRAIN_PATH`, and `YOUR_IMAGENET100_VAL_PATH`.
4. Run the script: ./run_imagenet100_class.sh

- For **Data-IL** experiments
3. Check run_imagenet100_data.sh and change `PROJECT`, `ENTITY`, `DATA_DIR`, `YOUR_IMAGENET100_TRAIN_PATH`, and `YOUR_IMAGENET100_VAL_PATH`.
4. Run the script: ./run_imagenet100_data.sh


### CSSL Experiments using DomainNet (Domain-IL)
1. Navigate to the project directory: cd PNR
2. Download [DomainNet dataset](http://ai.bu.edu/M3SDA/) and place it in `YOUR_DOMAINNET_PATH`.
3. Check run_domainnet.sh and change `PROJECT`, `ENTITY`, `DATA_DIR`, and `YOUR_DOMAINNET_PATH`
4. Run the script: ./run_domainnet.sh


### Acknolwedgement

This code is implemented based on the official code of [CaSSLe](https://github.com/DonkeyShot21/cassle) and I would like to show my sincere gratitude to authors of it.

-------
## Citation

@InProceedings{pmlr-v235-cha24a,
  title = 	 {Regularizing with Pseudo-Negatives for Continual Self-Supervised Learning},
  author =       {Cha, Sungmin and Cho, Kyunghyun and Moon, Taesup},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {6048--6065},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR}
}


-------