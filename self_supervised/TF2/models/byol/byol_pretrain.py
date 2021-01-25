import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam

from absl import app
import math
from datetime import datetime
import os

from self_supervised.TF2.models.networks.resnet import ResNet18, ResNet34, \
    ResNet50
from self_supervised.TF2.models.networks.vgg import VGG_UNet_Encoder, \
    VGG_UNet_Decoder

from self_supervised.TF2.utils.accelerator import setup_accelerator
from self_supervised.TF2.dataset.cifar10 import load_input_fn
from self_supervised.TF2.dataset.oai import load_dataset
from self_supervised.TF2.models.simclr.simclr_flags import FLAGS


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
        return 1 - (1 - self.init_tau) * (math.cos(math.pi * self.global_step) + 1) / 2

    @tf.function
    def update_weights(self):
        for online_layer, target_layer in zip(self.model.online_network.layers, self.model.target_network.layers):
            target_layer.set_weights(
                self.current_tau * target_layer.get_weights() + (1 - self.current_tau) * online_layer.get_weights())


def MLP(name, hidden_size=512, projection_size=128):
    """ MLP head for projector and predictor """
    model = tf.keras.Sequential(name=name)

    model.add(tfkl.Flatten())
    model.add(tfkl.Dense(hidden_size, use_bias=False))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.ReLU())
    model.add(tfkl.Dense(projection_size, use_bias=True))

    return model


def online_network(backbone, hidden_size, projection_size):

    x = backbone.output
    y = MLP('projection', hidden_size, projection_size)(x)
    z = MLP('prediction', hidden_size, projection_size)(y)

    return tf.keras.Model(
        inputs=backbone.input, outputs=[x, y, z], name='online_network')


def target_network(backbone, hidden_size, projection_size):

    x = backbone.output
    y = MLP('projection', hidden_size, projection_size)(x)

    return tf.keras.Model(
        inputs=backbone.input, outputs=[x, y], name='target_network')


class BYOL(tf.keras.Model):

    def __init__(self,
                 backbone,
                 hidden_size,
                 projection_size,
                 online_ft=False,
                 linear=False):
        super(BYOL, self).__init__()

        self.online_network = online_network(
            backbone, hidden_size, projection_size)
        backbone_2 = tf.keras.models.clone_model(backbone)
        self.target_network = target_network(
            backbone_2, hidden_size, projection_size)
        self.online_ft = online_ft
        if online_ft:
            if linear:
                self.classifier = tf.keras.Sequential(
                    [tfkl.Flatten(),
                     tfkl.Dense(10, activation='softmax')
                     ], name='classifier')
            else:
                self.classifier = tf.keras.Sequential(
                    [tfkl.Flatten(),
                     tfkl.Dense(512, use_bias=False),
                     tfkl.BatchNormalization(),
                     tfkl.ReLU(),
                     tfkl.Dense(10, activation='softmax')
                     ], name='classifier')

    def build(self, input_shape):
        self.online_network.build(input_shape)
        if self.online_ft:
            self.classifier.build(
                self.online_network.compute_output_shape(input_shape)[0])
        self.target_network.build(input_shape)

        self.built = True

    def call(self, x, training=False):
        x, _, _ = self.online_network(x, training=training)
        return x

    def compile(self, ft_optimizer=None, **kwargs):
        super(BYOL, self).compile(**kwargs)
        if self.online_ft:
            assert ft_optimizer is not None, \
                'ft_optimizer should not be None if self.online_ft is True'
            self.ft_optimizer = ft_optimizer

    def cosine_similarity(self, x, y):
        x = tf.linalg.l2_normalize(x, axis=-1)
        y = tf.linalg.l2_normalize(y, axis=-1)
        return 2 - 2*tf.math.reduce_mean(tf.math.reduce_sum(x * y, axis=-1))

    def compute_loss(self, data, training=False):
        x, y = data

        view_1 = x[..., :3]
        view_2 = x[..., 3:]

        _, _, online_out_1 = self.online_network(view_1, training=training)
        _, _, online_out_2 = self.online_network(view_2, training=training)
        _, target_out_1 = self.target_network(view_1, training=training)
        _, target_out_2 = self.target_network(view_2, training=training)

        loss = self.cosine_similarity(online_out_1, target_out_2)
        loss += self.cosine_similarity(online_out_2, target_out_1)

        return loss

    def train_step(self, data):
        # apply gradient tape to online network only
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data, training=True)
        trainable_variables = self.online_network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.online_ft:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            results = {'similarity_loss': loss, **metrics_results}
        else:
            results = {'similarity_loss': loss}

        return results

    def finetune_step(self, data):
        x, y = data
        view = x[..., :3]

        with tf.GradientTape() as tape:
            features = self(view, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.classifier.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

    def test_step(self, data):
        x, y = data
        view = x[..., :3]
        loss = self.compute_loss(data, training=False)
        if self.online_ft:
            features = self(view, training=False)
            y_pred = self.classifier(features, training=False)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metric_results}
        else:
            return {'similarity_loss': loss}


def main(argv):
    del argv

    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)

    # load datasets:
    if FLAGS.dataset == 'cifar10':
        train_ds = load_input_fn(split='train',
                                 batch_size=FLAGS.batch_size,
                                 training_mode='pretrain',
                                 normalize=FLAGS.normalize)

        val_ds = load_input_fn(split='test',
                               batch_size=FLAGS.batch_size,
                               training_mode='pretrain',
                               normalize=FLAGS.normalize)

        ds_info = tfds.builder(FLAGS.dataset).info
        steps_per_epoch = ds_info.splits['train'].num_examples // FLAGS.batch_size
        validation_steps = ds_info.splits['test'].num_examples // FLAGS.batch_size
        ds_shape = (32, 32, 3)
    elif FLAGS.dataset == 'oai_challenge':
        train_ds, val_ds = load_dataset(
            batch_size=FLAGS.batch_size,
            dataset_dir='gs://oai-challenge-dataset/tfrecords',
            training_mode='pretrain')
        steps_per_epoch = 19200 // FLAGS.batch_size
        validation_steps = 4480 // FLAGS.batch_size
        ds_shape = (288, 288, 1)

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

        model = BYOL(backbone=backbone,
                     hidden_size=1024,
                     projection_size=64,
                     online_ft=True,
                     linear=True)

        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(
                learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(
                lr=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adamw':
            optimizer = AdamW(
                weight_decay=1e-06, learning_rate=FLAGS.learning_rate)

        # build model and compile it
        model.compile(ft_optimizer=Adam(learning_rate=3e-03),
                      optimizer=optimizer,
                      loss=tf.keras.losses.sparse_categorical_crossentropy, 
                      metrics=['acc'])
        model.build((None, *ds_shape))
        model.online_network.summary()

    # Define checkpoint
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(FLAGS.logdir, time)

    movingavg_cb = BYOLMAWeightUpdate()

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=FLAGS.train_epochs,
              validation_data=val_ds,
              validation_steps=validation_steps,
              callbacks=[movingavg_cb])

    model.save_weights(os.path.join(logdir, 'byol_weights.hdf5'))


if __name__ == '__main__':
    app.run(main)
