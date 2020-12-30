import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1

from absl import app
from datetime import datetime
import os

from self_supervised.TF2.models.networks.resnet import ResNet50
from self_supervised.TF2.utils.accelerator import setup_accelerator
from self_supervised.TF2.utils.losses import nt_xent_loss
from self_supervised.TF2.dataset.cifar10 import load_input_fn
from self_supervised.TF2.models.simclr.simclr_flags import FLAGS


def projection_head(proj_head_dim=512,
                    output_dim=128,
                    num_layers=1
                    ):

    model = tf.keras.Sequential()
    model.add(tfkl.GlobalAveragePooling2D())
    model.add(tfkl.Flatten())

    for _ in range(num_layers):
        model.add(tfkl.Dense(proj_head_dim, use_bias=False))
        model.add(tfkl.BatchNormalization())
        model.add(tfkl.ReLU())

    model.add(tfkl.Dense(output_dim, use_bias=True))

    return model

class SimCLR(tf.keras.Model):

    def __init__(self,
                 backbone,
                 projection,
                 loss_temperature=0.5,
                 online_ft=False,
                 linear_ft=False):

        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection = projection
        self.loss_temperature = loss_temperature
        self.online_ft = online_ft

        if online_ft:
            if linear_ft:
                self.classifier = tf.keras.Sequential([
                            tfkl.Flatten(),
                            tfkl.Dense(10, activation='softmax')
                        ], name='classifier')
            else:
                self.classifier = tf.keras.Sequential([
                            tfkl.Flatten(),
                            tfkl.Dense(512, use_bias=False),
                            tfkl.BatchNormalization(),
                            tfkl.ReLU(),
                            tfkl.Dense(10, activation='softmax')
                        ], name='classifier')

    def build(self, input_shape):

        self.backbone.build(input_shape)
        if self.projection is not None:
            self.projection.build(self.backbone.compute_output_shape(input_shape))
        if self.online_ft:
            self.classifier.build(self.backbone.compute_output_shape(input_shape))

        self.built = True

    def call(self, x, training=False):

        result = self.backbone(x, training=training)

        return result

    def compile(self, loss_fn=nt_xent_loss, ft_optimizer=None, **kwargs):
        super(SimCLR, self).compile(**kwargs)
        self.loss_fn = loss_fn
        if self.online_ft:
            assert ft_optimizer is not None, 'ft_optimizer should not be None if self.online_ft is True'
            self.ft_optimizer = ft_optimizer

    def compute_output_shape(self, input_shape):

        current_shape = self.backbone.compute_output_shape(input_shape)
        if self.projection is not None:
            current_shape = self.projection.compute_output_shape(current_shape)
        return current_shape

    def shared_step(self, data, training):

        x, y = data
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
        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        if self.online_ft:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metrics_results}
        else:
            return {'similarity_loss': loss}

    def finetune_step(self, data):
        
        x, y = data
        num_channels = int(x.shape[-1] // 2)
        view = x[..., :num_channels]

        with tf.GradientTape() as tape:
            features = self.backbone(view, training=True)
            y_pred = self.classifier(features, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_variables = self.classifier.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

    def test_step(self, data):

        sim_loss = self.shared_step(data, training=False)
        if self.online_ft:
            x, y = data
            num_channels = int(x.shape[-1] // 2)
            view = x[..., :num_channels]
            features = self.backbone(view, training=False)
            y_pred = self.classifier(features, training=False)
            classifier_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': sim_loss, **metric_results}
        else:
            return {'loss': loss}

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

    with strategy.scope():
        # load model
        model = SimCLR(backbone=ResNet50(input_shape=ds_shape),
                       projection=projection_head(),
                       loss_temperature=0.5,
                       online_ft=True,
                       linear_ft=False)

        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(learning_rate=FLAGS.learning_rate,
                             weight_decay_rate=1e-04,
                             exclude_from_weight_decay=['bias', 'BatchNormalization'])
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'sgd':
            optimizer = SGD(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adamw':
            optimizer = AdamW(weight_decay=1e-06, learning_rate=FLAGS.learning_rate)

        model.compile(optimizer=optimizer,
                      loss_fn=nt_xent_loss,
                      ft_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-03),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['acc'])
  
        model.build((None, *ds_shape))
        model.backbone.summary()
        model.projection.summary()

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(FLAGS.logdir, time)

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=FLAGS.train_epochs,
              validation_data=val_ds,
              validation_steps=validation_steps,
              verbose=1)

    model.save_weights(os.path.join(logdir, 'simclr_weights.hdf5'))

if __name__ == '__main__':
    app.run(main)
