# Experiments to run CIFAR10 experiments
## batch size = 256
python3 -m autodidatta.models.simclr --histdir training_logs/CIFAR10/batch_size=256/SimCLR --logdir weights/CIFAR10/batch_size=256 --batch_size=256
python3 -m autodidatta.models.simsiam --histdir training_logs/CIFAR10/batch_size=256/SimSiam --logdir weights/CIFAR10/batch_size=256 --batch_size=256
python3 -m autodidatta.models.byol --histdir training_logs/CIFAR10/batch_size=256/BYOL --logdir weights/CIFAR10/batch_size=256 --batch_size=256
python3 -m autodidatta.models.barlow_twins --histdir training_logs/CIFAR10/batch_size=256/Barlow_Twins --logdir weights/CIFAR10/batch_size=256 --batch_size=256
