from absl import flags

# Dataset
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'BraTS', 'OAI'], 'cifar10 (default), BraTS, OAI')
flags.DEFINE_integer('image_size', 32, 'set image size for training and evaluation')
flags.DEFINE_string('dataset_dir', '.', 'set directory of your dataset')

# Training
flags.DEFINE_string('train_mode', 'pretrain', 'set whether to pretrain(default) or finetune')
flags.DEFINE_enum('optimizer', 'adam', ['lamb', 'adam', 'sgd'], 'lamb (default), adam')
flags.DEFINE_integer('batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 1e-02, 'set learning rate for optimizer.')
flags.DEFINE_float('weight_decay', 1e-04, 'set weight decay')

# Model specification
flags.DEFINE_enum('backbone', 'resnet50', ['resnet50', 'vgg16', 'vgg19'], 'resnet50 (default)')
flags.DEFINE_bool('use_2D', True, 'set whether to train on 2D or 3D data. Required for BraTS and OAI only')
flags.DEFINE_float('loss_temperature', 0.5, 'set temperature for loss function')
flags.DEFINE_bool('use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer('num_cores', 8, 'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string('tpu', 'oai-tpu', 'set the name of TPU device')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')

FLAGS = flags.FLAGS
