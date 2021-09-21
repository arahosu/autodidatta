from absl import app
from absl import flags
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger

from autodidatta.datasets.cifar10 import load_input_fn
from autodidatta.datasets.oai import load_dataset
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.vgg import VGG_UNet_Encoder, \
    VGG_UNet_Decoder, build_unet
from autodidatta.utils.accelerator import setup_accelerator
from autodidatta.utils.loss import tversky_loss
from autodidatta.utils.metrics import DiceMetrics

# Dataset
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'stl10', 'imagenet'],
    'cifar10 (default), oai, stl10, imagenet')
flags.DEFINE_string(
    'dataset_dir', 'gs://oai-challenge-dataset/tfrecords',
    'directory for where the dataset is stored')

# Training
flags.DEFINE_string(
    'pretrain_weights', None,
    'Directory for the pre-trained model weights.')
flags.DEFINE_integer(
    'train_epochs', 1000, 'Number of epochs to train the model')
flags.DEFINE_float(
    'fraction_data',
    1.0,
    'fraction of training data to be used during downstream evaluation')
flags.DEFINE_enum(
    'optimizer', 'adam', ['lamb', 'adam', 'sgd', 'adamw'],
    'optimizer for pre-training')
flags.DEFINE_integer('batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 1e-03, 'set learning rate for optimizer.')

# Classification task args
flags.DEFINE_bool(
    'eval_linear', True,
    'Set whether to run linear (Default) or non-linear evaluation protocol')

# Segmentation task args
flags.DEFINE_bool(
    'finetune_decoder_only', False,
    'Set whether to finetune only the decoder during training'
)
flags.DEFINE_bool(
    'multiclass', True,
    'Set whether to train multi-class (Default) or binary segmentation model')
flags.DEFINE_bool(
    'add_background', True,
    'Set whether to include background class (Default)'
)

# Model specification args
flags.DEFINE_enum(
    'backbone', 'resnet18',
    ['resnet50', 'resnet34', 'resnet18', 'vgg_uent'],
    'resnet50 (default), resnet18, resnet34', 'vgg_unet')

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
flags.DEFINE_bool(
    'use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer(
    'num_cores', 8, 'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string('tpu', 'local', 'set the name of TPU device')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')

FLAGS = flags.FLAGS


def main(argv):

    del argv

    # Step 1: Select whether to train on single/multi-GPU/TPU
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)

    # Step 2: Select dataset
    if FLAGS.dataset == 'cifar10':
        train_ds = load_input_fn(
            is_training=True,
            batch_size=FLAGS.batch_size,
            image_size=32,
            pre_train=False)
        validation_ds = load_input_fn(
            is_training=False,
            batch_size=FLAGS.batch_size,
            image_size=32,
            pre_train=False)
        ds_shape = (32, 32, 3)

        ds_info = tfds.builder(FLAGS.dataset).info
        num_train_examples = ds_info.splits['train'].num_examples
        num_val_examples = ds_info.splits['test'].num_examples
        steps_per_epoch = num_train_examples // FLAGS.batch_size
        validation_steps = num_val_examples // FLAGS.batch_size

    elif FLAGS.datasets == 'oai':

        train_ds, validation_ds = load_dataset(
            FLAGS.dataset_dir,
            FLAGS.batch_size,
            288,
            19200,
            'finetune',
            FLAGS.fraction_data,
            FLAGS.multi_class,
            FLAGS.add_background,
            True)

        ds_shape = (288, 288, 1)
        num_train_examples = 19200
        num_val_examples = 4480
        steps_per_epoch = num_train_examples // FLAGS.batch_size
        validation_steps = num_val_examples // FLAGS.batch_size

        if FLAGS.multi_class:
            if FLAGS.add_background:
                num_classes = 7
            else:
                num_classes = 6
        else:
            num_classes = 1

    # Step 3: Define model to be trained
    with strategy.scope():
        # load model
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)
        elif FLAGS.backbone == 'vgg_unet':
            backbone = VGG_UNet_Encoder(input_shape=ds_shape)

        backbone.load_weights(FLAGS.pretrain_weights)
        if FLAGS.finetune_decoder_only:
            backbone.trainable = False

        if FLAGS.backbone == 'vgg_unet':
            model = build_unet(encoder=backbone,
                               decoder=VGG_UNet_Decoder,
                               num_classes=num_classes)
        else:
            if FLAGS.eval_linear:
                classifier = tf.keras.Sequential(
                    [tfkl.Flatten(),
                        tfkl.Dense(10, activation='softmax')],
                    name='classifier')
            else:
                classifier = tf.keras.Sequential(
                    [tfkl.Flatten(),
                        tfkl.Dense(512, use_bias=False),
                        tfkl.BatchNormalization(),
                        tfkl.ReLU(),
                        tfkl.Dense(10, activation='softmax')],
                    name='classifier')

            model = tf.keras.Sequential(
                [backbone,
                    classifier]
            )

        # select optimizer
        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(
                learning_rate=FLAGS.learning_rate,
                weight_decay_rate=1e-04,
                exclude_from_weight_decay=['bias', 'BatchNormalization'])
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'sgd':
            optimizer = SGD(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adamw':
            optimizer = AdamW(
                weight_decay=1e-06, learning_rate=FLAGS.learning_rate)

        if FLAGS.dataset == 'oai':
            loss_fn = tversky_loss
            metrics = [DiceMetrics(idx=idx) for idx in range(num_classes)]
        else:
            loss_fn = tf.keras.losses.sparse_categorical_crossentropy
            metrics = ['acc']

        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=metrics)

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = None

    if FLAGS.save_weights:
        logdir = os.path.join(FLAGS.logdir, time)
        os.mkdir(logdir)
    if FLAGS.save_history:
        histdir = os.path.join(FLAGS.histdir, time)
        os.mkdir(histdir)

        # Create a callback for saving the training results into a csv file
        histfile = 'simsiam_results.csv'
        csv_logger = CSVLogger(os.path.join(histdir, histfile))
        cb = [csv_logger]

        # Save flag params in a flag file in the same subdirectory
        flagfile = os.path.join(histdir, 'train_flags.cfg')
        FLAGS.append_flags_into_file(flagfile)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.train_epochs,
        validation_data=validation_ds,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=cb)

    if FLAGS.save_weights:
        weights_name = 'simsiam_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name))


if __name__ == '__main__':
    app.run(main)
