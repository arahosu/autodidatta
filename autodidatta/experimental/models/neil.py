from absl import app
from absl import flags
from datetime import datetime
import math
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger

import autodidatta.augment as A
from autodidatta.datasets import cifar10, stl10
from autodidatta.flags import dataset_flags, training_flags, utils_flags
from autodidatta.models.base import BaseModel
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.mlp import projection_head, predictor_head
from autodidatta.utils.accelerator import setup_accelerator

# Redefine default value
flags.FLAGS.set_default(
    'proj_hidden_dim', 64)
flags.FLAGS.set_default(
    'output_dim', 1)
flags.FLAGS.set_default(
    'use_bfloat16', False)

FLAGS = flags.FLAGS

class NEIL(BaseModel):
    
    def __init__(self,
                 backbone,
                 projector,
                 predictor,
                 geometric_aug_fn,
                 transform_aug_fn,
                 classifier=None):
        
        super(NEIL, self).__init__(
            backbone=backbone,
            projector=projector,
            predictor=predictor,
            classifier=classifier
        )

        self.geometric_aug_fn = geometric_aug_fn
        self.transform_aug_fn = transform_aug_fn

    def shared_step(self, data, training):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data 
        
        xi = self.transform_aug_fn(x, training=training)
        zi = self.backbone(xi, training=training)
        yi = self.projector(zi, training=training)

        y_pred, xj = self.geometric_aug_fn(yi, x, training=training)

        zj = self.backbone(xj, training=training)
        yj = self.projector(zj, training=training)

        loss = self.loss_fn(yj, y_pred)
        loss /= self.distribute_strategy.num_replicas_in_sync
        return loss
    
    def finetune_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            features = self(x, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.classifier.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
    
    def test_step(self, data):

        if isinstance(data, tuple):
            x, y = data
        else:
            x = data

        loss = self.shared_step(data, training=False)

        if self.classifier is not None:
            features = self.backbone(x, training=False)
            y_pred = self.classifier(features, training=False)
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metric_results}
        else:
            return {'similarity_loss': loss}


def conv_projector(hidden_filters,
                   output_filters,
                   num_layers=1,
                   batch_norm_output=False,
                   global_bn=True):
    
    model = tf.keras.Sequential()
    model.add(tfkl.UpSampling2D(
        size=(8, 8), interpolation='bilinear'))
    
    for _ in range(num_layers):
        model.add(tfkl.Conv2D(hidden_filters, 3, use_bias=False, padding='same'))
        if global_bn:
            model.add(tfkl.experimental.SyncBatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5))
        else:
            model.add(tfkl.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5))
        model.add(tfkl.ReLU())

    model.add(
        tfkl.Conv2D(output_filters, 3, use_bias=not batch_norm_output, padding='same'))
    if batch_norm_output:
        if global_bn:
            model.add(tfkl.experimental.SyncBatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5))
        else:
            model.add(tfkl.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5))
    
    return model


def main(argv):

    del argv

    # Choose accelerator 
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)
    
    # Choose whether to train with float32 or bfloat16 precision
    if FLAGS.use_bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Select dataset
    ds_info = tfds.builder(FLAGS.dataset).info
    if FLAGS.dataset == 'cifar10':
        load_dataset = cifar10.load_input_fn
        image_size = 32
        train_split = 'train'
        validation_split = 'test'
    elif FLAGS.dataset == 'stl10':
        load_dataset = stl10.load_input_fn
        image_size = 96
        train_split = 'unlabelled'
        validation_split = 'test'
    else:
        raise NotImplementedError("other datasets have not yet been implmented")
    
    num_train_examples = ds_info.splits[train_split].num_examples
    num_val_examples = ds_info.splits[validation_split].num_examples
    steps_per_epoch = num_train_examples // FLAGS.batch_size
    validation_steps = num_val_examples // FLAGS.batch_size
    ds_shape = (image_size, image_size, 3)

    # Define train-validation split
    train_ds = load_dataset(
        is_training=True,
        batch_size=FLAGS.batch_size,
        image_size=image_size,
        aug_fn=None,
        pre_train=False,
        use_bfloat16=FLAGS.use_bfloat16)
    validation_ds = load_dataset(
        is_training=False,
        batch_size=FLAGS.batch_size,
        image_size=image_size,
        aug_fn=None,
        pre_train=False,
        use_bfloat16=FLAGS.use_bfloat16)

    with strategy.scope():
        # Define augmentation functions
        geometric_aug_fn = A.Augment([
            A.layers.RandomResizedCrop(
                image_size, image_size, scale=(0.08, 1.0)),
            A.layers.HorizontalFlip(p=0.5)
        ])  
        transform_aug_fn = A.Augment([
            A.layers.ColorJitter(
                FLAGS.brightness,
                FLAGS.contrast,
                FLAGS.saturation,
                FLAGS.hue,
                p=0.8),
            A.layers.ToGray(p=0.2)
        ])

        # Define backbone
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)
        else:
            raise NotImplementedError("other backbones have not yet been implemented")

        # If online finetuning is enabled
        if FLAGS.online_ft:
            assert FLAGS.dataset != 'stl10', \
                'Online finetuning is not supported for stl10'

            # load classifier for downstream task evaluation
            classifier = training_flags.load_classifier()

            finetune_loss = tf.keras.losses.sparse_categorical_crossentropy
            metrics = ['acc']
        else:
            classifier = None

        model = NEIL(
            backbone=backbone,
            projector=conv_projector(
                FLAGS.proj_hidden_dim,
                FLAGS.output_dim),
            predictor=None,
            geometric_aug_fn=geometric_aug_fn,
            transform_aug_fn=transform_aug_fn,
            classifier=classifier)

        # load_optimizer
        optimizer, ft_optimizer = training_flags.load_optimizer(num_train_examples)

        if FLAGS.online_ft:
            model.compile(
                optimizer=optimizer,
                loss_fn=tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.NONE),
                ft_optimizer=ft_optimizer,
                loss=finetune_loss,
                metrics=metrics)
        else:
            model.compile(
                optimizer=optimizer,
                loss_fn=tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.NONE))

        # Build the model
        model.build((None, *ds_shape))

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
        histfile = 'neil_results.csv'
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
        weights_name = 'neil_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name),
                           save_backbone_only=True)


if __name__ == '__main__':
    app.run(main)
