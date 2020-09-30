import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam

from absl import app
import math
from datetime import datetime
import os

from self_supervised.TF2.models.networks.resnet50 import ResNet50
from self_supervised.TF2.utils.accelerator import setup_accelerator
from self_supervised.TF2.models.simclr.simclr_flags import FLAGS

class BYOLMAWeightUpdate(tf.keras.callbacks.Callback):

    def __init__(self, maxsteps, init_tau=0.996):
        super(BYOLMAWeightUpdate, self).__init__()

        self.maxsteps = maxsteps
        self.init_tau = init_tau
        self.current_tau = init_tau
        self.global_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.global_step += 1

        self.update_weights()
        self.current_tau = self.update_tau()

    def update_tau(self):
        return 1 - (1 - self.init_tau) * (math.cos(math.pi * self.global_step / self.maxsteps) + 1) / 2

    def update_weights(self):
        for online_module, target_module in zip(self.model.online_network.layers, self.model.target_network.layers):
            if hasattr(online_module, 'layers'):
                for online_layer, target_layer in zip(online_module.layers, target_module.layers):
                    if hasattr(online_layer, 'kernel'):
                        target_layer.kernel = self.current_tau * target_layer.kernel + (1 - self.current_tau) * online_layer.kernel
                    if hasattr(online_layer, 'bias') and online_layer.bias is not None:
                        target_layer.bias = self.current_tau * target_layer.bias + (1 - self.current_tau) * online_layer.bias

def MLP(name, hidden_size=4096, projection_size=256):
    """ MLP head for projector and predictor """
    model = tf.keras.Sequential(name=name)

    model.add(tfkl.Flatten())
    model.add(tfkl.Dense(hidden_size, use_bias=False))
    model.add(tfkl.BatchNormalization())
    model.add(tfkl.ReLU())
    model.add(tfkl.Dense(projection_size, use_bias=True))

    return model

class SiameseArm(tf.keras.Model):

    def __init__(self, name, in_shape, hidden_size=4096, projection_size=256):
        super(SiameseArm, self).__init__(name=name)

        self.backbone = ResNet50(include_top=False,
                                 input_shape=in_shape)
        self.projection = MLP(name="projection",
                              hidden_size=hidden_size,
                              projection_size=projection_size)
        self.prediction = MLP(name="prediction",
                              hidden_size=hidden_size,
                              projection_size=projection_size)

    def call(self, x, training=False):

        y = self.backbone(x, training=training)
        z = self.projection(y, training=training)
        h = self.prediction(z, training=training)

        return y, z, h

class BYOL(tf.keras.Model):

    def __init__(self, in_shape, hidden_size, projection_size):
        super(BYOL, self).__init__()

        self.online_network = SiameseArm(name="online_network",
                                         in_shape=in_shape,
                                         hidden_size=hidden_size,
                                         projection_size=projection_size)
        self.target_network = SiameseArm(name="target_network",
                                         in_shape=in_shape,
                                         hidden_size=hidden_size,
                                         projection_size=projection_size)

    def build(self, input_shape):
        self.online_network.build(input_shape)
        self.target_network.build(input_shape)
        self.built = True

    def call(self, x, training=False):
        y, _, _ = self.online_network(x, training=training)
        return y

    def compile(self, optimizer, **kwargs):
        super(BYOL, self).compile(**kwargs)
        self.optimizer = optimizer

    def cosine_similarity(self, x, y):
        x = tf.linalg.l2_normalize(x, axis=-1)
        y = tf.linalg.l2_normalize(y, axis=-1)
        return tf.math.reduce_mean(tf.math.reduce_sum(x * y, axis=-1))

    def compute_loss(self, data):
        x, y = data

        view_1 = x[..., :3]
        view_2 = x[..., 3:]

        _, _, online_out_1 = self.online_network(view_1, training=True)
        _, _, online_out_2 = self.online_network(view_2, training=True)
        _, target_out_1, _ = self.target_network(view_1, training=True)
        _, target_out_2, _ = self.target_network(view_2, training=True)

        loss = -2 * self.cosine_similarity(online_out_1, target_out_2)
        loss += -2 * self.cosine_similarity(online_out_2, target_out_1)

        return loss

    def save_weights(self, filepath):
        self.online_network.save_weights(filepath=filepath)

    def train_step(self, data):
        # apply gradient tape to online network only
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)
        trainable_variables = self.online_network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        return {'train_loss': loss}

    def test_step(self, data):
        pass

def main(argv):
    del argv

    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)

    if FLAGS.dataset == 'cifar10':
        from self_supervised.TF2.dataset.cifar10 import load_input_fn
        import tensorflow_datasets as tfds

        train_ds = load_input_fn(split=tfds.Split.TRAIN,
                                 name='cifar10',
                                 batch_size=FLAGS.batch_size,
                                 training_mode='pretrain')

        ds_info = tfds.builder(FLAGS.dataset).info
        steps_per_epoch = ds_info.splits['train'].num_examples // FLAGS.batch_size
        ds_shape = (32, 32, 3)

    with strategy.scope():

        model = BYOL(in_shape=ds_shape,
                     hidden_size=512,
                     projection_size=128)

        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(lr=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adamw':
            optimizer = AdamW(weight_decay=1e-06, learning_rate=FLAGS.learning_rate)

        # build model and compile it
        model.compile(optimizer=optimizer)
        model.build((None, *ds_shape))
        model.summary()

    # Define checkpoint
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(FLAGS.logdir, time)

    movingavg_cb = BYOLMAWeightUpdate(maxsteps=FLAGS.train_epochs * steps_per_epoch)

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=FLAGS.train_epochs,
              callbacks=[movingavg_cb])

    model.save_weights(os.path.join(logdir, 'byol_weights.ckpt'))

if __name__ == '__main__':
    app.run(main)
