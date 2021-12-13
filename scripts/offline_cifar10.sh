# BYOL Offline Evaluation
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-053705/byol_weights.hdf5 --percentage_data=100 --histdir ./training_logs/CIFAR10/offline/BYOL/
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-053705/byol_weights.hdf5 --percentage_data=10 --histdir ./training_logs/CIFAR10/offline/BYOL/
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-053705/byol_weights.hdf5 --percentage_data=1 --histdir ./training_logs/CIFAR10/offline/BYOL/

# SimCLR Offline Evaluation
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211203-234505/simclr_weights.hdf5 --percentage_data=100 --histdir ./training_logs/CIFAR10/offline/SimCLR/
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211203-234505/simclr_weights.hdf5 --percentage_data=10 --histdir ./training_logs/CIFAR10/offline/SimCLR/
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211203-234505/simclr_weights.hdf5 --percentage_data=1 --histdir ./training_logs/CIFAR10/offline/SimCLR/

# SimSiam Offline Evaluation
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-023951/simsiam_weights.hdf5 --histdir ./training_logs/CIFAR10/offline/SimSiam/ --percentage_data=100
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-023951/simsiam_weights.hdf5 --histdir ./training_logs/CIFAR10/offline/SimSiam/ --percentage_data=10
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-023951/simsiam_weights.hdf5 --histdir ./training_logs/CIFAR10/offline/SimSiam/ --percentage_data=1

# Barlow Twins Offline Evaluation
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-215835/barlow_twins_weights.hdf5 --histdir ./training_logs/CIFAR10/offline/Barlow_Twins/ --percentage_data=100
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-215835/barlow_twins_weights.hdf5 --histdir ./training_logs/CIFAR10/offline/Barlow_Twins/ --percentage_data=10
python3 -m autodidatta.models.finetune --weights=weights/CIFAR10/batch_size=256/20211204-215835/barlow_twins_weights.hdf5 --histdir ./training_logs/CIFAR10/offline/Barlow_Twins/ --percentage_data=1