from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import tensorflow as tf

from autodidatta.augment import load_aug_fn_pretrain
from autodidatta.datasets import Dataset
from autodidatta.models import get_model_cls, load_classifier
from autodidatta.utils.accelerator import setup_accelerator
from autodidatta.utils.optimizers import load_optimizer
from autodidatta.utils.callbacks import load_callbacks

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('configs')

def main(_):
    # Set random seed
    tf.random.set_seed(FLAGS.configs.seed)

    # Setup GPU/TPU
    strategy= setup_accelerator(**FLAGS.configs.accelerator_configs)
    
    # Set dtype policy
    tf.keras.mixed_precision.set_global_policy(
        FLAGS.configs.dtype_str)

    # Load datasets
    dataset = Dataset(FLAGS.configs.dataset_name,
                      FLAGS.configs.train_split,
                      FLAGS.configs.eval_split,
                      FLAGS.configs.dataset_dir)

    # Load augmentation functions
    aug_1, aug_2, eval_aug = load_aug_fn_pretrain(
        FLAGS.configs.dataset_name,
        FLAGS.configs.model,
        dataset.ds_shape[0],
        FLAGS.configs.aug_configs,
        FLAGS.configs.seed)
    
    train_ds, eval_ds = dataset.load_pretrain_datasets(
        batch_size=FLAGS.configs.batch_size,
        eval_batch_size=FLAGS.configs.eval_batch_size,
        train_aug=aug_1,
        eval_aug=eval_aug,
        train_aug_2=aug_2,
        seed=FLAGS.configs.seed,
        dtype_policy=FLAGS.configs.dtype_str)

    steps_per_epoch = dataset.num_train_examples // FLAGS.configs.batch_size
    eval_steps = dataset.num_eval_examples // FLAGS.configs.eval_batch_size
    max_steps = int(dataset.num_train_examples // FLAGS.configs.batch_size) \
                * FLAGS.configs.train_epochs

    # Load model
    with strategy.scope():
        if FLAGS.configs.online_ft:
            classifier = load_classifier(
                dataset.num_classes,
                FLAGS.configs.linear_eval,
                strategy)
        
        model = get_model_cls(
            dataset.ds_shape,
            FLAGS.configs.model,
            FLAGS.configs.model_configs,
            classifier=classifier)
        
        optimizer = load_optimizer(
            optimizer_name=FLAGS.configs.optimizer,
            learning_rate=FLAGS.configs.learning_rate,
            num_train_examples=dataset.num_train_examples,
            batch_size=FLAGS.configs.batch_size,
            train_epochs=FLAGS.configs.train_epochs,
            warmup_epochs=FLAGS.configs.warmup_epochs,
            optimizer_configs=FLAGS.configs.optimizer_configs)
        
        ft_optimizer = load_optimizer(
            optimizer_name=FLAGS.configs.ft_optimizer,
            learning_rate=FLAGS.configs.ft_learning_rate,
            num_train_examples=dataset.num_train_examples,
            batch_size=FLAGS.configs.batch_size,
            train_epochs=FLAGS.configs.train_epochs,
            warmup_epochs=FLAGS.configs.warmup_epochs,
            optimizer_configs=FLAGS.configs.ft_optimizer_configs)

        finetune_loss = tf.keras.losses.sparse_categorical_crossentropy
        model.compile(
            ft_optimizer=ft_optimizer,
            optimizer=optimizer, loss=finetune_loss, metrics=['acc'])
        model.build((None, *dataset.ds_shape))
        model.summary()

    # Load callbacks
    cb = load_callbacks(
        FLAGS.configs.model,
        FLAGS.configs.history_dir,
        FLAGS.configs.weights_dir,
        FLAGS.configs.weights_filename,
        FLAGS.configs.history_filename,
        FLAGS.configs.online_ft,
        max_steps,
        FLAGS.configs.callback_configs)

    # Train model
    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.configs.train_epochs,
        validation_data=eval_ds,
        validation_steps=eval_steps,
        verbose=1,
        callbacks=cb)

if __name__ == '__main__':
    app.run(main)