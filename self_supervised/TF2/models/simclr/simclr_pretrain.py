import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1

from absl import app
from datetime import datetime
import os

from self_supervised.TF2.models.networks.resnet50 import ResNet50
from self_supervised.TF2.utils.accelerator import setup_accelerator
from self_supervised.TF2.utils.losses import nt_xent_loss, nt_xent_loss_v2
from self_supervised.TF2.dataset.cifar10 import load_input_fn
from self_supervised.TF2.models.simclr.simclr_flags import FLAGS


def get_projection_head(use_2D=True,
                        proj_head_dim=512,
                        output_dim=128
                        ):

    model = tf.keras.Sequential()

    if use_2D:
        model.add(tfkl.GlobalAveragePooling2D())
    else:
        model.add(tfkl.GlobalAveragePooling3D())

    model.add(tfkl.Flatten())
    model.add(tfkl.Dense(proj_head_dim, use_bias=True))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.ReLU())

    model.add(tfkl.Dense(output_dim, use_bias=False))

    return model

class SimCLR(tf.keras.Model):

    def __init__(self,
                 backbone,
                 projection,
                 loss_temperature=0.5):

        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection = projection
        self.loss_temperature = loss_temperature

    def build(self, input_shape):

        self.backbone.build(input_shape)
        if self.projection is not None:
            self.projection.build(self.backbone.compute_output_shape(input_shape))

        self.built = True

    def call(self, x, training=False):

        result = self.backbone(x, training=training)

        return result

    def compile(self, optimizer, loss_fn=nt_xent_loss):
        super(SimCLR, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def compute_output_shape(self, input_shape):

        current_shape = self.backbone.compute_output_shape(input_shape)
        if self.projection is not None:
            current_shape = self.projection.compute_output_shape(current_shape)
        return current_shape

    def shared_step(self, data, training):

        x, y = data

        xi = x[..., :3]
        xj = x[..., 3:]

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

        return {'train_loss': loss}

    def test_step(self, data):

        loss = self.shared_step(data, training=False)

        return {'_loss': loss}

def main(argv):

    del argv

    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)
    global_batch_size = FLAGS.num_cores * FLAGS.batch_size

    # load datasets:
    if FLAGS.dataset == 'cifar10':
        train_ds = load_input_fn(split=tfds.Split.TRAIN,
                                 name='cifar10',
                                 batch_size=FLAGS.batch_size,
                                 training_mode='pretrain',
                                 use_cloud=False if FLAGS.use_gpu else True,
                                 normalize=FLAGS.normalize)

        val_ds = load_input_fn(split=tfds.Split.TEST,
                               name='cifar10',
                               batch_size=FLAGS.batch_size,
                               training_mode='pretrain',
                               use_cloud=False if FLAGS.use_gpu else True,
                               normalize=FLAGS.normalize)

        ds_info = tfds.builder(FLAGS.dataset).info
        steps_per_epoch = ds_info.splits['train'].num_examples // global_batch_size
        validation_steps = ds_info.splits['test'].num_examples // global_batch_size
        ds_shape = (32, 32, 3)

    with strategy.scope():
        # load model
        backbone = ResNet50(include_top=False,
                            input_shape=ds_shape,
                            pooling=None)

        model = SimCLR(backbone=backbone,
                       projection=get_projection_head(),
                       loss_temperature=FLAGS.loss_temperature)

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

        model.compile(optimizer=optimizer, loss_fn=nt_xent_loss)
        model.build((None, *ds_shape))
        model.backbone.summary()
        model.projection.summary()

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(FLAGS.logdir, time)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(logdir + '/simclr_weights.{epoch:03d}.ckpt',
                                                 save_best_only=False,
                                                 save_weights_only=True)

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              batch_size=global_batch_size,
              epochs=1000,
              validation_data=val_ds,
              validation_steps=validation_steps,
              verbose=1)

    model.save_weights(logdir + '/simclr_weights.ckpt')


if __name__ == '__main__':
    app.run(main)
