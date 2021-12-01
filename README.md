# AutoDiDatta

AutoDiDatta (ADD) is a Python library of self-supervised learning methods for unsupervised representation learning, powered by the Keras API and Tensorflow 2. The main goal of this library is to provide a set of high-quality implementations of SOTA self-supervised learning methods. 

ADD implements some of the most popular self-supervised learning methods, including

- [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
- [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)
- [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf)
- [BYOL](https://arxiv.org/pdf/2006.07733.pdf)

## Results

### CIFAR10

| Method       | Backbone | FT (1%) | FT (10%) | FT (100%) | Linear Eval (online) |
|--------------|----------|---------|----------|-----------|----------------------|
| Barlow Twins | ResNet18 |         |          |           | 90.22                |
| BYOL         | ResNet18 |         |          |           | 91.63                |
| SimCLR       | ResNet18 |         |          |           | 90.88                |
| SimSiam      | ResNet18 |         |          |           | 90.18                |

### CIFAR100

| Method       | Backbone | FT (1%) | FT (10%) | FT (100%) | Linear Eval (online) |
|--------------|----------|---------|----------|-----------|----------------------|
| Barlow Twins | ResNet18 |         |          |           |                      |
| BYOL         | ResNet18 |         |          |           | 68.28                |
| SimCLR       | ResNet18 |         |          |           | 66.39                |
| SimSiam      | ResNet18 |         |          |           | 64.41                |
