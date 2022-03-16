# AutoDiDatta

AutoDiDatta (ADD) is a Python library of self-supervised learning methods for unsupervised representation learning, powered by the Keras API and Tensorflow 2. The main goal of this library is to provide a set of high-quality implementations of SOTA self-supervised learning methods. 

ADD implements some of the most popular self-supervised learning methods, including

- [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
- [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)
- [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf)
- [BYOL](https://arxiv.org/pdf/2006.07733.pdf)

## Running the experiments

### Requirements
Dependencies (Python >= 3.7)

```{bash}
tensorflow==2.8.0
tensorflow-addons==0.16.1	
tensorflow_datasets
ml_collections
```

### Model training
Pre-training with online lienar evaluation:

```{bash}
# SimCLR pre-training on CIFAR-10 dataset
python3 -m examples.pretrain --configs=examples/configs/CIFAR10/simclr_cifar10_config.py

# SimCLR pre-training on CIFAR-10 dataset (no online linear eval)
python3 -m examples.pretrain --configs=examples/configs/CIFAR10/simclr_cifar10_config.py --online_ft=False
```

Offline linear evaluation on pre-trained model backbone:
```{bash}
# SimCLR offline linear evaluation on CIFAR-10 dataset, replace MODEL_WEIGHTS_DIR with your saved model weights
python3 -m examples.finetune --configs=examples/configs/CIFAR10/simclr_cifar10_finetune.py --weights=MODEL_WEIGHTS

# SimCLR finetuning on CIFAR-10 dataset
python3 -m examples.finetune --configs=examples/configs/CIFAR10/simclr_cifar10_finetune.py --weights=MODEL_WEIGHTS --finetune=True
```

You can also specify training split to perform linear evaluation using a fraction of training labels (i.e. 10%)
```{bash}
# SimCLR offline linear evaluation using 10% of training labels
python3 -m examples.finetune --configs=examples/configs/CIFAR10/simclr_cifar10_finetune.py --weights=MODEL_WEIGHTS --train_split='train[:10%]'
```

## Linear Evaluation Results

### CIFAR10

|    Method    | Top-1 Acc. (online) | Top-1 Acc. (offline) |
|:------------:|:-------------------:|----------------------|
| Barlow Twins |        90.82        |         90.43        |
| BYOL         |        91.55        |         91.79        |
| SimCLR       |        90.37        |         90.84        |
| SimSiam      |        89.37        |         89.61        |

### CIFAR100

|    Method    | Top-1 Acc. (online) | Top-1 Acc. (offline) |
|:------------:|:-------------------:|----------------------|
| Barlow Twins |        66.17        |         67.60        |
| BYOL         |        68.01        |         68.28        |
| SimCLR       |        66.16        |         66.39        |
| SimSiam      |        62.34        |         62.36        |

## Semi-Supervised Evaluation Results

### CIFAR10

|    Method    |   1%  | 10%   |
|:------------:|:-----:|-------|
| Barlow Twins | 85.70 | 89.24 |
| BYOL         | 84.66 | 90.04 |
| SimCLR       | 84.39 | 89.24 |
| SimSiam      | 84.51 | 87.95 |
| Supervised   | 38.52 | 73.97 |

### CIFAR100

|    Method    |   1%  | 10%   |
|:------------:|:-----:|-------|
| Barlow Twins | 39.89 | 59.19 |
| BYOL         | 36.31 | 58.93 |
| SimCLR       | 34.86 | 58.05 |
| SimSiam      | 32.89 | 51.74 |
| Supervised   | 8.45  | 23.07 |
