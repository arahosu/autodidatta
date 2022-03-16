from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import tensorflow as tf
import tensorflow.keras.layers as tfkl

import autodidatta.augment as A
from autodidatta.datasets import Dataset
from autodidatta.models import get_backbone_only
from autodidatta.utils.accelerator import setup_accelerator
from autodidatta.utils.optimizers import load_optimizer
from autodidatta.utils.callbacks import load_callbacks

# Random seed
flags.DEFINE_integer(
    'seed', 42,
    'random seed')

# Dataset flags
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'cifar100', 'stl10', 'imagenet2012'],
    'cifar10 (default), cifar100, stl10, imagenet2012')
flags.DEFINE_string(
    'dataset_dir', None,
    'directory where the dataset is stored')
flags.DEFINE_string(
    'train_split', 'train',
    'string for the training split of the dataset')
flags.DEFINE_string(
    'eval_split', 'test',
    'string for the evaluation split of the dataset')

# Accelerator flags
flags.DEFINE_bool(
    'use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer(
    'num_cores', 8,
    'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string(
    'device_name', 'local', 
    'set the name of GPU/TPU device')

# Training flags
flags.DEFINE_integer(
    'batch_size', 256, 'set batch size for training')
flags.DEFINE_integer(
    'train_epochs', 200, 'Number of epochs to finetune the model')
flags.DEFINE_enum('dtype_str', 'mixed_bfloat16',
    ['mixed_bfloat16', 'mixed_float16', 'float32'],
    'set global policy for dtype')
flags.DEFINE_string(
    'weights', None,
    'Directory where pre-trained model weights are saved')
flags_DEFINE_bool(
    'finetune', False,
    'Set whether to finetune the whole model (default) or perform linear eval')

# Logging
flags.DEFINE_string(
    'save_dir', 'examples/log/finetune',
    'Tensorboard logging dir')


config_flags.DEFINE_config_file('configs')

FLAGS = flags.FLAGS

def main(_):
    # Set random seed
    tf.random.set_seed(FLAGS.seed)

    # Setup GPU/TPU
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.device_name)
    
    # Set dtype policy
    tf.keras.mixed_precision.set_global_policy(
        FLAGS.dtype_str)

    # Load datasets
    dataset = Dataset(FLAGS.dataset,
                      FLAGS.train_split,
                      FLAGS.eval_split,
                      FLAGS.dataset_dir)

    # Load augmentation functions
    train_aug = tf.keras.Sequential(
        [
            A.layers.RandomResizedCrop(
                dataset.ds_shape[0], dataset.ds_shape[1], scale=(0.08, 1.0)),
            A.layers.HorizontalFlip(
                p=0.5),
            A.layers.Normalize(
                mean=FLAGS.configs.mean,
                std=FLAGS.configs.std)
        ]
    )

    eval_aug = tf.keras.Sequential()
    eval_aug.add(A.layers.Normalize(
        mean=FLAGS.configs.mean,
        std=FLAGS.configs.std))
    
    if FLAGS.dataset == 'imagenet2012':
        eval_aug.add(A.layers.CentralCrop(224, 224, 0.875))
    
    # Load datasets 
    train_ds, eval_ds = dataset.load_finetune_datasets(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        train_aug=train_aug,
        eval_aug=eval_aug,
        seed=FLAGS.seed)

    steps_per_epoch = dataset.num_train_examples // FLAGS.batch_size
    eval_steps = dataset.num_eval_examples // FLAGS.batch_size
    max_steps = steps_per_epoch * FLAGS.train_epochs

    # Load model
    with strategy.scope():
        classifier = tf.keras.Sequential(
            [tfkl.Flatten(),
             tfkl.Dense(
                 dataset.num_classes,
                 activation='softmax')],
            name='classifier')
        
        backbone = get_backbone_only(
            dataset.ds_shape,
            FLAGS.configs.backbone)
        if not FLAGS.finetune:
            backbone.trainable = False
        backbone.load_weights(FLAGS.weights)
    
        model = tf.keras.Sequential(
            [backbone, classifier])
        
        optimizer = load_optimizer(
            optimizer_name=FLAGS.configs.optimizer,
            learning_rate=FLAGS.configs.learning_rate,
            optimizer_configs=FLAGS.configs.optimizer_configs)

        finetune_loss = tf.keras.losses.sparse_categorical_crossentropy
        model.compile(
            optimizer=optimizer, loss=finetune_loss, metrics=['acc'])
        model.build((None, *dataset.ds_shape))
        model.summary()

    # Load callbacks
    cb = load_callbacks(
        model_name=FLAGS.configs.model,
        log_dir=FLAGS.save_dir,
        weights_dir=FLAGS.save_dir,
        online_ft=True,
        max_steps=max_steps,
        callback_configs=FLAGS.configs.callback_configs)

    # Train model
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.train_epochs,
        validation_data=eval_ds,
        validation_steps=eval_steps,
        verbose=1,
        callbacks=cb)

if __name__ == '__main__':
    flags.mark_flag_as_required('configs')
    flags.mark_flag_as_required('weights')
    app.run(main)