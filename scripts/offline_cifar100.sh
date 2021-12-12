# # BYOL Offline Evaluation
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-054503/byol_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/BYOL/ --percentage_data=100 --dataset=cifar100
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-054503/byol_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/BYOL/ --percentage_data=10 --dataset=cifar100
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-054503/byol_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/BYOL/ --percentage_data=1 --dataset=cifar100

# # SimCLR Offline Evaluation
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211203-235250/simclr_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/SimCLR/ --percentage_data=100 --dataset=cifar100
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211203-235250/simclr_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/SimCLR/ --percentage_data=10 --dataset=cifar100
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211203-235250/simclr_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/SimCLR/ --percentage_data=1 --dataset=cifar100

# SimSiam Offline Evaluation
python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-024757/simsiam_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/SimSiam/ --percentage_data=100 --dataset=cifar100
python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-024757/simsiam_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/SimSiam/ --percentage_data=10 --dataset=cifar100
python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-024757/simsiam_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/SimSiam/ --percentage_data=1 --dataset=cifar100

# Barlow Twins Offline Evaluation
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-220629/barlow_twins_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/Barlow_Twins/ --percentage_data=100 --dataset=cifar100
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-220629/barlow_twins_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/Barlow_Twins/ --percentage_data=10 --dataset=cifar100
# python3 -m autodidatta.models.finetune --weights=weights/CIFAR100/batch_size=256/20211204-220629/barlow_twins_weights.hdf5 --histdir ./training_logs/CIFAR100/offline/Barlow_Twins/ --percentage_data=1 --dataset=cifar100