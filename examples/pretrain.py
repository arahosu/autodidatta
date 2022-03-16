from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from autodidatta.augment import load_aug_fn_pretrain
from autodidatta.datasets import Dataset
from autodidatta.models import get_model_cls
from autodidatta.utils.accelerator import setup_accelerator
from autodidatta.utils.optimizers import load_optimizer, WarmUpAndCosineDecay
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
    'batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_integer(
    'eval_batch_size', 256, 'set batch size for evaluation')
flags.DEFINE_integer(
    'train_epochs', 1000, 'Number of epochs to train the model')
flags.DEFINE_enum('dtype_str', 'mixed_bfloat16',
    ['mixed_bfloat16', 'mixed_float16', 'float32'],
    'set global policy for dtype')

# Online finetuning flags
flags.DEFINE_bool(
    'online_ft', True,
    'set whether to enable online finetuning (True by default)')

# Logging
flags.DEFINE_string(
    'save_dir', 'examples/log',
    'Tensorboard logging dir')
flags.DEFINE_string(
    'weights_dir', 'examples/log',
    'Directory where weights are saved')

# Model-specific configs
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
    aug_1, aug_2, eval_aug = load_aug_fn_pretrain(
        FLAGS.dataset,
        dataset.ds_shape[0],
        FLAGS.configs.aug_configs,
        FLAGS.seed)
    
    train_ds, eval_ds = dataset.load_pretrain_datasets(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        train_aug=aug_1,
        eval_aug=eval_aug,
        train_aug_2=aug_2,
        seed=FLAGS.seed,
        dtype_policy=FLAGS.dtype_str)

    steps_per_epoch = dataset.num_train_examples // FLAGS.batch_size
    eval_steps = dataset.num_eval_examples // FLAGS.eval_batch_size
    max_steps = steps_per_epoch * FLAGS.train_epochs

    # Load model
    with strategy.scope():
        if FLAGS.online_ft:
            classifier = tf.keras.Sequential(
                [tfkl.Flatten(),
                 tfkl.Dense(
                     dataset.num_classes,
                     activation='softmax')],
                 name='classifier')
        
        model = get_model_cls(
            dataset.ds_shape,
            FLAGS.configs.model,
            FLAGS.configs.model_configs,
            classifier=classifier)
        
        lr_schedule = WarmUpAndCosineDecay(
            base_learning_rate=FLAGS.configs.base_learning_rate,
            num_examples=dataset.num_train_examples,
            batch_size=FLAGS.batch_size,
            num_train_epochs=FLAGS.train_epochs,
            **FLAGS.configs.scheduler_configs)

        optimizer = load_optimizer(
            optimizer_name=FLAGS.configs.optimizer,
            learning_rate=lr_schedule,
            optimizer_configs=FLAGS.configs.optimizer_configs)
        
        ft_optimizer = load_optimizer(
            optimizer_name=FLAGS.configs.ft_optimizer,
            learning_rate=FLAGS.configs.ft_learning_rate,
            optimizer_configs=FLAGS.configs.ft_optimizer_configs)

        finetune_loss = tf.keras.losses.sparse_categorical_crossentropy
        model.compile(
            ft_optimizer=ft_optimizer,
            optimizer=optimizer, loss=finetune_loss, metrics=['acc'])
        model.build((None, *dataset.ds_shape))
        model.summary()

    # Load callbacks
    cb = load_callbacks(
        model_name=FLAGS.configs.model,
        log_dir=FLAGS.save_dir,
        weights_dir=FLAGS.save_dir,
        online_ft=FLAGS.online_ft,
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
    app.run(main)