# To train Barlow Twins / BYOL / SimCLR / SimSiam on a GPU, run the following command
# python3 -m autodidatta.models.[the method that you want to run] --use_gpu --num_cores=1
# Replace the square bracket with barlow_twins, byol, simclr, simsiam, e.t.c


# To save your training history and model weights in a directory, add the following arguments
# python3 -m autodidatta.models.[the method that you want to run] --use_gpu --num_cores=1 --histdir=[directory for training history] --logdir=[directory for saved weights]


# To evaluate your model in a offline setting, run the following script
# python3 -m autodidatta.models.finetune --use_gpu --num_cores=1 --weights=[file for your saved weights] --percentage_data=100
# Percentage data flag defines the percentage of images in the training set that you want to train your model on. 