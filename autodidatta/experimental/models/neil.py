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
from autodidatta.experimental.datasets.oai import load_dataset
from autodidatta.experimental.models.vgg import VGG_UNet
from autodidatta.experimental.utils.loss import invariance_variance_loss, tversky_loss
from autodidatta.experimental.utils.metrics import dice_coef
from autodidatta.flags import dataset_flags, training_flags, utils_flags
from autodidatta.models.base import BaseModel
from autodidatta.utils.accelerator import setup_accelerator

# Redefine default value
flags.FLAGS.set_default(
    'ft_learning_rate', 1e-03)
flags.FLAGS.set_default(
    'warmup_epochs', 0)
flags.FLAGS.set_default(
    'brightness', 0.4)
flags.FLAGS.set_default(
    'contrast', 0.4)
flags.FLAGS.set_default(
    'proj_hidden_dim', 64)
flags.FLAGS.set_default(
    'pred_hidden_dim', 64)
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
        
        # Non-geometric transformation
        xi = self.transform_aug_fn(x, training=training)
        feat_i = self.backbone(xi, training=training)
        zi = self.projector(feat_i, training=training)
        pi = self.predictor(zi, training=training)

        concat_input = tf.concat([zi, pi, x], axis=-1)
        concat_output = self.geometric_aug_fn(concat_input, training=training)

        z_pred = concat_output[..., :int(zi.shape[-1])]
        p_pred = concat_output[..., int(zi.shape[-1]):int(zi.shape[-1] + pi.shape[-1])]
        xj = concat_output[..., int(zi.shape[-1] + pi.shape[-1]):]
        
        feat_j = self.backbone(xj, training=training)
        zj = self.projector(feat_j, training=training)
        pj = self.predictor(zj, training=training)

        loss = self.loss_fn(tf.stop_gradient(z_pred), pj) / 2
        loss += self.loss_fn(tf.stop_gradient(zj), p_pred) / 2
        return loss
    
    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        
        if self.predictor is not None:
            trainable_variables = self.backbone.trainable_variables + \
                self.projector.trainable_variables + \
                self.predictor.trainable_variables
        else:
            trainable_variables = self.backbone.trainable_variables + \
                self.projector.trainable_variables

        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.classifier is not None:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            results = {'similarity_loss': loss, **metrics_results}
        else:
            results = {'similarity_loss': loss}

        return results
    
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
    
    for _ in range(num_layers):
        model.add(tfkl.Conv2D(hidden_filters, 1, use_bias=False, padding='same'))
        if global_bn:
            model.add(tfkl.experimental.SyncBatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5))
        else:
            model.add(tfkl.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5))
        model.add(tfkl.ReLU())

    model.add(
        tfkl.Conv2D(output_filters, 1, use_bias=not batch_norm_output, padding='same'))
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

    tf.config.set_soft_device_placement(True)

    # Choose accelerator 
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)
    
    # Choose whether to train with float32 or bfloat16 precision
    if FLAGS.use_bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Select dataset
    train_ds, validation_ds = load_dataset(
        'gs://oai-challenge-dataset/tfrecords',
        batch_size=FLAGS.batch_size,
        image_size=288,
        training_mode='finetune',
        fraction_data=1.0,
        multi_class=False,
        add_background=False,
        normalize=True)

    num_train_examples = 19200
    num_val_examples = 4480
    steps_per_epoch = num_train_examples // FLAGS.batch_size
    validation_steps = num_val_examples // FLAGS.batch_size
    ds_shape = (288, 288, 1)

    with strategy.scope():
        # Define augmentation functions
        geometric_aug_fn = tf.keras.Sequential(
            [tfkl.RandomZoom((0.65, -0.65)),
             tfkl.RandomFlip()
            ]
        ) 
        transform_aug_fn = A.Augment([
            A.layers.RandomBrightness(FLAGS.brightness),
            A.layers.RandomContrast(FLAGS.contrast),
            A.layers.RandomGamma([0.5, 1.5], gain=1.0)
        ])

        # Define backbone
        backbone = VGG_UNet(ds_shape)

        # If online finetuning is enabled
        if FLAGS.online_ft:

            # load classifier for downstream task evaluation
            classifier = tf.keras.Sequential(
                [tfkl.Conv2D(1, 1, activation='sigmoid', padding='same')]
            )

            finetune_loss = tversky_loss
            metrics = [dice_coef]
        else:
            classifier = None

        model = NEIL(
            backbone=backbone,
            projector=conv_projector(
                FLAGS.proj_hidden_dim,
                FLAGS.output_dim),
            predictor=conv_projector(
                FLAGS.pred_hidden_dim,
                FLAGS.output_dim),
            geometric_aug_fn=geometric_aug_fn,
            transform_aug_fn=transform_aug_fn,
            classifier=classifier)

        # load_optimizer
        optimizer, ft_optimizer = training_flags.load_optimizer(num_train_examples)

        if FLAGS.online_ft:
            model.compile(
                optimizer=optimizer,
                loss_fn=tf.keras.losses.cosine_similarity,
                ft_optimizer=ft_optimizer,
                loss=finetune_loss,
                metrics=metrics)
        else:
            model.compile(
                optimizer=optimizer,
                loss_fn=tf.keras.losses.cosine_similarity)

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
