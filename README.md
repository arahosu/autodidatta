# AutoDiDatta

AutoDiDatta (ADD) is a Python library of self-supervised learning methods for unsupervised representation learning, powered by the Keras API and Tensorflow 2. The main goal of this library is to provide a set of high-quality implementations of SOTA self-supervised learning methods. 

ADD implements some of the most popular self-supervised learning methods, including

- [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
- [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)
- [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf)
- [BYOL](https://arxiv.org/pdf/2006.07733.pdf)

## Linear Evaluation Results

### CIFAR10

| Method       | Backbone | FT (1%) | FT (10%) | FT (100%) | Linear Eval (online) |
|--------------|----------|---------|----------|-----------|----------------------|
| Barlow Twins | ResNet18 | 85.70   | 89.24    | 90.43     | 90.82                |
| BYOL         | ResNet18 | 84.66   | 90.04    | 91.55     | 91.79                |
| SimCLR       | ResNet18 | 84.39   | 89.24    | 90.37     | 90.84                |
| SimSiam      | ResNet18 | 84.51   | 87.95    | 89.37     | 89.61                |

### CIFAR100

| Method       | Backbone | FT (1%) | FT (10%) | FT (100%) | Linear Eval (online) |
|--------------|----------|---------|----------|-----------|----------------------|
| Barlow Twins | ResNet18 | 39.89   | 59.19    | 66.17     | 67.60                |
| BYOL         | ResNet18 | 36.31   | 58.93    | 68.01     | 68.28                |
| SimCLR       | ResNet18 | 34.86   | 58.05    | 66.16     | 66.39                |
| SimSiam      | ResNet18 | 32.89   | 51.74    | 62.34     | 62.36                |
