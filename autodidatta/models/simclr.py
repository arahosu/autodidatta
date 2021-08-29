from absl import app
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger

from autodidatta.datasets.cifar10 import load_input_fn
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.mlp import projection_head
from autodidatta.utils.loss import nt_xent_loss
from autodidatta.utils.accelerator import setup_accelerator
from autodidatta.models.simclr_flags import FLAGS


class SimCLR(tf.keras.Model):

    def __init__(self,
                 backbone,
                 projection,
                 classifier=None,
                 loss_temperature=0.5):

        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection = projection
        self.classifier = classifier
        self.loss_temperature = loss_temperature

    def build(self, input_shape):

        self.backbone.build(input_shape)
        if self.projection is not None:
            self.projection.build(
                self.backbone.compute_output_shape(input_shape))
        if self.classifier is not None:
            self.classifier.build(
                self.backbone.compute_output_shape(input_shape))

        self.built = True

    def call(self, x, training=False):

        result = self.backbone(x, training=training)

        return result

    def compile(self, loss_fn=nt_xent_loss, ft_optimizer=None, **kwargs):
        super(SimCLR, self).compile(**kwargs)
        self.loss_fn = loss_fn
        if self.classifier is not None:
            assert ft_optimizer is not None, \
                'ft_optimizer should not be None if self.classifier is not \
                    None'
            self.ft_optimizer = ft_optimizer

    def compute_output_shape(self, input_shape):

        current_shape = self.backbone.compute_output_shape(input_shape)
        if self.projection is not None:
            current_shape = self.projection.compute_output_shape(current_shape)
        return current_shape

    def shared_step(self, data, training):

        x, _ = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        zi = self.backbone(xi, training=training)
        zj = self.backbone(xj, training=training)

        if self.projection is not None:
            zi = self.projection(zi, training=training)
            zj = self.projection(zj, training=training)

        zi = tf.math.l2_normalize(zi, axis=-1)
        zj = tf.math.l2_normalize(zj, axis=-1)

        loss = self.loss_fn(zi, zj, self.loss_temperature)

        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        trainable_variables = self.backbone.trainable_variables + \
            self.projection.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.classifier is not None:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metrics_results}
        else:
            return {'similarity_loss': loss}

    def finetune_step(self, data):

        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]
        if len(y.shape) > 2:
            num_classes = int(y.shape[-1] // 2)
            y = y[..., :num_classes]

        with tf.GradientTape() as tape:
            features = self.backbone(view, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.classifier.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

    def test_step(self, data):

        sim_loss = self.shared_step(data, training=False)
        if self.classifier is not None:
            x, y = data
            num_channels = int(x.shape[-1] // 2)
            view = x[..., :num_channels]
            if len(y.shape) > 2:
                num_classes = int(y.shape[-1] // 2)
                y = y[..., :num_classes]
            features = self.backbone(view, training=False)
            y_pred = self.classifier(features, training=False)
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': sim_loss, **metric_results}
        else:
            return {'loss': sim_loss}


def main(argv):

    del argv

    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)

    if FLAGS.dataset == 'cifar10':
        train_ds = load_input_fn(
            is_training=True,
            batch_size=FLAGS.batch_size,
            image_size=32,
            pre_train=True)
        validation_ds = load_input_fn(
            is_training=False,
            batch_size=FLAGS.batch_size,
            image_size=32,
            pre_train=True)
        ds_shape = (32, 32, 3)

    ds_info = tfds.builder(FLAGS.dataset).info
    num_train_examples = ds_info.splits['train'].num_examples
    num_val_examples = ds_info.splits['test'].num_examples
    steps_per_epoch = num_train_examples // FLAGS.batch_size
    validation_steps = num_val_examples // FLAGS.batch_size

    with strategy.scope():
        # load model
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)

        # If online finetuning is enabled
        if FLAGS.online_ft:
            # load model for downstream task evaluation
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

            loss = tf.keras.losses.sparse_categorical_crossentropy
            metrics = ['acc']
        else:
            classifier = None

        model = SimCLR(
            backbone=backbone,
            projection=projection_head(
                hidden_dim=FLAGS.hidden_dim,
                output_dim=FLAGS.output_dim,
                num_layers=FLAGS.num_head_layers,
                batch_norm_output=False),
            classifier=classifier,
            loss_temperature=FLAGS.loss_temperature)

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

        if classifier is not None:
            model.compile(
                optimizer=optimizer,
                loss_fn=nt_xent_loss,
                ft_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=FLAGS.ft_learning_rate),
                loss=loss,
                metrics=metrics)
        else:
            model.compile(optimizer=optimizer,
                          loss_fn=nt_xent_loss)

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
        histfile = 'simclr_results.csv'
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
        weights_name = 'simclr_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name))


if __name__ == '__main__':
    app.run(main)
