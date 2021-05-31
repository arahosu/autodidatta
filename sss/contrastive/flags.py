from absl import flags

# Dataset
flags.DEFINE_enum(
    'dataset', 'oai', ['cifar10', 'brats', 'oai'],
    'cifar10 (default), BraTS, oai')
flags.DEFINE_integer(
    'image_size',
    288,
    'Image size to be used.')
flags.DEFINE_string('dataset_dir', '.', 'set directory of your dataset')
flags.DEFINE_bool('normalize', True, 'set whether to normalize the images')

# Training
flags.DEFINE_integer(
    'train_epochs', 80, 'Number of epochs to train the model')
flags.DEFINE_enum(
    'optimizer', 'adam', ['lamb', 'adam', 'sgd', 'adamw'],
    'optimizer for pre-training')
flags.DEFINE_float(
    'loss_temperature', 0.5, 'set temperature for loss function')
flags.DEFINE_integer('batch_size', 64, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 2e-04, 'set learning rate for optimizer.')
flags.DEFINE_integer(
    'hidden_dim', 512,
    'set number of units in the hidden \
     layers of the projection/predictor head')
flags.DEFINE_integer(
    'output_dim', 128,
    'set number of units in the output layer of the projection/predictor head')
flags.DEFINE_integer(
    'num_head_layers', 1,
    'set number of intermediate layers in the projection head')
flags.DEFINE_bool(
    'eval_linear', True,
    'Set whether to run linear (Default) or non-linear evaluation protocol')
flags.DEFINE_bool(
    'custom_schedule', True,
    'Set whether to enable custom exponential decaying learning rate schedule')
flags.DEFINE_bool(
    'multi_class', False,
    'Set whether to train in a binary or multi-class/label (Default) settting')
flags.DEFINE_bool(
    'add_background', True,
    'Set whether to add background (multi-class) or not (multi-label)')

# Finetuning
flags.DEFINE_float(
    'fraction_data',
    1.0,
    'fraction of training data to be used during downstream evaluation'
)
flags.DEFINE_bool(
    'online_ft',
    True,
    'set whether to enable online finetuning (True by default)')
flags.DEFINE_float(
    'ft_learning_rate', 2e-04, 'set learning rate for finetuning optimizer')

# Model specification
flags.DEFINE_enum(
    'model',
    'simclr',
    ['simclr', 'simsiam', 'byol', 'barlow_twins', 'supervised'],
    'contrastive model to be trained')

# Classification-specific args
flags.DEFINE_enum(
    'backbone', 'resnet18',
    ['resnet50', 'resnet34', 'resnet18'],
    'resnet50 (default), resnet18, resnet34')

# Segmentation-specific args
flags.DEFINE_bool(
    'pretrain_encoder_only',
    True,
    'whether to pretrain encoder only for UNet'
)

# logging specification
flags.DEFINE_bool(
    'save_weights', False,
    'Whether to save weights. If True, weights are saved in logdir')
flags.DEFINE_bool(
    'save_history', True,
    'Whether to save the training history.'
)
flags.DEFINE_string(
    'histdir', '/home/User/Self-Supervised-Segmentation/training_logs',
    'Directory for where the training history is being saved'
)
flags.DEFINE_string(
    'logdir', '/home/User/Self-Supervised-Segmentation/weights',
    'Directory for where the weights are being saved')
flags.DEFINE_string(
    'weights', None,
    'Directory for the trained model weights. Only used for finetuning')
flags.DEFINE_bool(
    'use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer(
    'num_cores', 8, 'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string('tpu', 'oai-tpu', 'set the name of TPU device')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')

FLAGS = flags.FLAGS
