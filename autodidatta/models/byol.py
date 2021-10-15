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

from autodidatta.datasets.cifar10 import load_input_fn
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.mlp import projection_head, predictor_head
from autodidatta.utils.loss import byol_loss
from autodidatta.utils.accelerator import setup_accelerator

# Dataset
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'stl10', 'imagenet'],
    'cifar10 (default), stl10, imagenet')

# Training
flags.DEFINE_integer(
    'train_epochs', 1000, 'Number of epochs to train the model')
flags.DEFINE_enum(
    'optimizer', 'adam', ['lamb', 'adam', 'sgd', 'adamw'],
    'optimizer for pre-training')
flags.DEFINE_float(
    'init_tau', 0.99, 'initial tau parameter for target network update')
flags.DEFINE_integer('batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 1e-03, 'set learning rate for optimizer.')
flags.DEFINE_integer(
    'hidden_dim', 4096,
    'set number of units in the hidden \
     layers of the projection/predictor head')
flags.DEFINE_integer(
    'output_dim', 256,
    'set number of units in the output layer of the projection/predictor head')
flags.DEFINE_integer(
    'num_head_layers', 1,
    'set number of intermediate layers in the projection head')
flags.DEFINE_bool(
    'eval_linear', True,
    'Set whether to run linear (Default) or non-linear evaluation protocol')

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

# Model specification args
flags.DEFINE_enum(
    'backbone', 'resnet18',
    ['resnet50', 'resnet34', 'resnet18'],
    'resnet50 (default), resnet18, resnet34')

# logging specification
flags.DEFINE_bool(
    'save_weights', False,
    'Whether to save weights. If True, weights are saved in logdir')
flags.DEFINE_bool(
    'save_history', True,
    'Whether to save the training history.'
)
flags.DEFINE_string(
    'histdir', '/home/User/autodidatta/training_logs',
    'Directory for where the training history is being saved'
)
flags.DEFINE_string(
    'logdir', '/home/User/autodidatta/weights',
    'Directory for where the weights are being saved')
flags.DEFINE_string(
    'weights', None,
    'Directory for the trained model weights. Only used for finetuning')
flags.DEFINE_bool(
    'use_gpu', 'False', 'set whether to use GPU')
flags.DEFINE_integer(
    'num_cores', 8, 'set number of cores/workers for TPUs/GPUs')
flags.DEFINE_string('tpu', 'local', 'set the name of TPU device')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')

FLAGS = flags.FLAGS


class BYOLMAWeightUpdate(tf.keras.callbacks.Callback):

    def __init__(self, init_tau=0.99):
        super(BYOLMAWeightUpdate, self).__init__()

        self.init_tau = init_tau
        self.current_tau = init_tau
        self.global_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1
        self.update_weights()
        self.current_tau = self.update_tau()

    def update_tau(self):
        return 1 - (1 - self.init_tau) * \
            (math.cos(math.pi * self.global_step) + 1) / 2

    @tf.function
    def update_weights(self):
        for online_layer, target_layer in zip(
                self.model.online_network.layers,
                self.model.target_network.layers):
            if all(hasattr(target_layer, attr) for attr in ["kernel", "bias"]):
                target_layer.kernel.assign(self.current_tau *
                                           target_layer.kernel
                                           + (1 - self.current_tau) *
                                           online_layer.kernel)
                target_layer.bias.assign(self.current_tau * target_layer.bias
                                         + (1 - self.current_tau) *
                                         online_layer.bias)


class BYOL(tf.keras.Model):

    def __init__(self,
                 online_network,
                 target_network,
                 classifier=None):

        super(BYOL, self).__init__()

        self.online_network = online_network
        self.target_network = target_network
        self.classifier = classifier

    def build(self, input_shape):

        self.online_network.build(input_shape)

        if self.classifier is not None:
            self.classifier.build(
                self.online_network.compute_output_shape(input_shape)[0])
        self.target_network.build(input_shape)

        self.built = True

    def call(self, x, training=False):

        x, _, _ = self.online_network(x, training=training)

        return x

    def compile(self, loss_fn, ft_optimizer=None, **kwargs):
        super(BYOL, self).compile(**kwargs)
        self.loss_fn = loss_fn
        if self.classifier is not None:
            assert ft_optimizer is not None, \
                'ft_optimizer should not be None if self.classifier is not \
                    None'
            self.ft_optimizer = ft_optimizer

    def shared_step(self, data, training):

        x, _ = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        _, _, zi = self.online_network(xi, training=training)
        _, _, zj = self.online_network(xj, training=training)
        _, pi = self.target_network(xi, training=training)
        _, pj = self.target_network(xj, training=training)

        loss = self.loss_fn(tf.stop_gradient(pi), zj)
        loss += self.loss_fn(tf.stop_gradient(pj), zi)

        return loss

    def finetune_step(self, data):
        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        if len(y.shape) > 2:
            num_classes = int(y.shape[-1] // 2)
            y = y[..., :num_classes]

        with tf.GradientTape() as tape:
            features = self(view, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.classifier.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        trainable_variables = self.online_network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.classifier is not None:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            results = {'similarity_loss': loss, **metrics_results}
        else:
            results = {'similarity_loss': loss}

        return results

    def test_step(self, data):
        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        if len(y.shape) > 2:
            num_classes = int(y.shape[-1] // 2)
            y = y[..., :num_classes]
        loss = self.shared_step(data, training=False)

        if self.classifier is not None:
            features = self(view, training=False)
            y_pred = self.classifier(features, training=False)
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metric_results}
        else:
            return {'similarity_loss': loss}


def build_online_network(backbone, hidden_dim, output_dim):

    x = backbone.output
    y = projection_head(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        batch_norm_output=True)(x)
    z = predictor_head(hidden_dim=hidden_dim, output_dim=output_dim)(y)

    return tf.keras.Model(
        inputs=backbone.input, outputs=[x, y, z], name='online_network')


def build_target_network(backbone, hidden_dim, output_dim):

    x = backbone.output
    y = projection_head(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        batch_norm_output=True)(x)

    return tf.keras.Model(
        inputs=backbone.input, outputs=[x, y], name='target_network')


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

        online_network = build_online_network(
            backbone, FLAGS.hidden_dim, FLAGS.output_dim)
        backbone_2 = tf.keras.models.clone_model(backbone)
        target_network = build_target_network(
            backbone_2, FLAGS.hidden_dim, FLAGS.output_dim)

        model = BYOL(
            online_network=online_network,
            target_network=target_network,
            classifier=classifier
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

        if classifier is not None:
            model.compile(
                optimizer=optimizer,
                loss_fn=byol_loss,
                ft_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=FLAGS.ft_learning_rate),
                loss=loss,
                metrics=metrics)
        else:
            model.compile(optimizer=optimizer,
                          loss_fn=byol_loss)

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Moving Average Weight Update Callback
    movingavg_cb = BYOLMAWeightUpdate(init_tau=FLAGS.init_tau)
    cb = [movingavg_cb]
    # cb = []

    if FLAGS.save_weights:
        logdir = os.path.join(FLAGS.logdir, time)
        os.mkdir(logdir)
    if FLAGS.save_history:
        histdir = os.path.join(FLAGS.histdir, time)
        os.mkdir(histdir)

        # Create a callback for saving the training results into a csv file
        histfile = 'byol_results.csv'
        csv_logger = CSVLogger(os.path.join(histdir, histfile))

        cb.append(csv_logger)

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
        weights_name = 'byol_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name))


if __name__ == '__main__':
    app.run(main)
