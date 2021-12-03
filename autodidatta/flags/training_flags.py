from absl import flags
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW

from autodidatta.utils.optimizers import WarmUpAndCosineDecay

# Accelerator flags
flags.DEFINE_bool(
    'use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer(
    'num_cores', 8, 'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string('tpu', 'local', 'set the name of TPU device')

# Training flags
flags.DEFINE_enum(
    'backbone', 'resnet18',
    ['resnet50', 'resnet34', 'resnet18'],
    'resnet18 (default), resnet34, resnet50')
flags.DEFINE_integer(
    'batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_float(
    'learning_rate', 1e-03, 'set learning rate for optimizer.')
flags.DEFINE_float(
    'loss_temperature', 0.2, 'set temperature for loss function')
flags.DEFINE_integer(
    'num_head_layers', 1,
    'set number of intermediate layers in the projector/predictor head')
flags.DEFINE_enum(
    'optimizer', 'adamw', ['lamb', 'adam', 'sgd', 'adamw'],
    'optimizer for pre-training')
flags.DEFINE_integer(
    'output_dim', 512,
    'set number of units in the output layer of the projector/predictor head')
flags.DEFINE_integer(
    'pred_hidden_dim', 512,
    'set number of units in the hidden \
     layers of the projection head')
flags.DEFINE_integer(
    'proj_hidden_dim', 2048,
    'set number of units in the hidden \
     layers of the projector head')
flags.DEFINE_integer(
    'train_epochs', 1000, 'Number of epochs to train the model')
flags.DEFINE_bool(
    'train_projector', True,
    'Set whether to train the projector head or not (Default)')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')
flags.DEFINE_integer(
    'warmup_epochs', 10,
    'number of warmup epochs for learning rate scheduler')

# Finetuning flags
flags.DEFINE_bool(
    'eval_linear', True,
    'Set whether to run linear (Default) or non-linear evaluation protocol')
flags.DEFINE_float(
    'ft_learning_rate', 2e-04, 'set learning rate for finetuning optimizer')
flags.DEFINE_bool(
    'online_ft',
    True,
    'set whether to enable online finetuning (True by default)')

FLAGS = flags.FLAGS      

def load_optimizer(num_train_examples):
    
    lr_schedule = WarmUpAndCosineDecay(
                FLAGS.learning_rate, num_train_examples,
                FLAGS.batch_size, FLAGS.warmup_epochs, FLAGS.train_epochs,
                learning_rate_scaling='linear' if FLAGS.optimizer == 'sgd' else None)
    ft_lr_schedule = WarmUpAndCosineDecay(
                FLAGS.ft_learning_rate, num_train_examples,
                FLAGS.batch_size, FLAGS.warmup_epochs, FLAGS.train_epochs,
                learning_rate_scaling='linear' if FLAGS.optimizer == 'sgd' else None)

    if FLAGS.optimizer == 'lamb':
        optimizer = LAMB(
            learning_rate=lr_schedule,
            weight_decay_rate=1e-06,
            exclude_from_weight_decay=['bias', 'BatchNormalization'])
        ft_optimizer = LAMB(
            learning_rate=ft_lr_schedule,
            weight_decay_rate=1e-06,
            exclude_from_weight_decay=['bias', 'BatchNormalization'])
    elif FLAGS.optimizer == 'adam':
        optimizer = Adam(learning_rate=lr_schedule)
        ft_optimizer = Adam(learning_rate=ft_lr_schedule)
    elif FLAGS.optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
        ft_optimizer = SGD(
            learning_rate=ft_lr_schedule, momentum=0.9, nesterov=False)
    elif FLAGS.optimizer == 'adamw':
        optimizer = AdamW(
            weight_decay=1e-06, learning_rate=lr_schedule)
        ft_optimizer = AdamW(
            weight_decay=1e-06, learning_rate=ft_lr_schedule)
    else:
        raise NotImplementedError("other optimizers have not yet been implemented")

    return optimizer, ft_optimizer


def load_classifier(num_classes):

    if FLAGS.eval_linear:
        classifier = classifier = tf.keras.Sequential(
            [tfkl.Flatten(),
             tfkl.Dense(num_classes, activation='softmax')],
             name='classifier')
    else:
        if not FLAGS.use_gpu or FLAGS.num_cores <= 1:
            BatchNorm = tfkl.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5)
        else:
            BatchNorm = tfkl.experimental.SyncBatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5)

        classifier = tf.keras.Sequential(
            [tfkl.Flatten(),
             tfkl.Dense(512, use_bias=False),
             BatchNorm,
             tfkl.ReLU(),
             tfkl.Dense(num_classes, activation='softmax')],
             name='classifier')
        
    return classifier