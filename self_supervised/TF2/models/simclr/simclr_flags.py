from absl import flags

# Dataset
flags.DEFINE_enum('dataset', 'oai', ['cifar10', 'brats', 'oai'], 'cifar10 (default), BraTS, oai')
flags.DEFINE_integer('image_size', 288, 'set image size for training and evaluation')
flags.DEFINE_string('dataset_dir', '.', 'set directory of your dataset')
flags.DEFINE_bool('normalize', False, 'set whether to normalize the images')

# Training
flags.DEFINE_string('train_mode', 'pretrain', 'set whether to pretrain(default) or finetune')
flags.DEFINE_integer('train_epochs', 1000, 'Number of epochs to train the model')
flags.DEFINE_enum('optimizer', 'adamw', ['lamb', 'adam', 'sgd', 'adamw'], 'optimizer for pre-training')
flags.DEFINE_integer('batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 1e-03, 'set learning rate for optimizer.')
flags.DEFINE_float('weight_decay', 1e-04, 'set weight decay')
flags.DEFINE_integer('proj_head_dim', 512, 'set number of units in the intermediate layers of the projection head')
flags.DEFINE_integer('num_head_layers', 1, 'set number of intermediate layers in the projection head')

# Finetuning
flags.DEFINE_bool('eval_linear', True, 'Set whether to run linear (Default) or non-linear evaluation protocol')
flags.DEFINE_integer('eval_epochs', 200, 'Number of epochs to finetune the model')

# Model specification
flags.DEFINE_enum('backbone', 'resnet18', ['resnet50', 'resnet34', 'resnet18'], 'resnet50 (default)')
flags.DEFINE_bool('use_2D', True, 'set whether to train on 2D or 3D data. Required for BraTS and OAI only')
flags.DEFINE_string('logdir', 'gs://oai-challenge-dataset/weights/', 'Directory for where the weights are being saved')
flags.DEFINE_string('weights', None, 'Directory for the trained model weights. Only used for finetuning')
flags.DEFINE_float('loss_temperature', 0.5, 'set temperature for loss function')
flags.DEFINE_bool('use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer('num_cores', 8, 'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string('tpu', 'oai-tpu', 'set the name of TPU device')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')

FLAGS = flags.FLAGS
