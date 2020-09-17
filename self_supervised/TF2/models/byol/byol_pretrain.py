# python3 -m Self-Supervised-Segmentation.self_supervised.TF2.models.byol.byol_pretrain

import tensorflow as tf
# import tensorflow.keras.layers as tfkl
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.optimizers import Adam
import numpy as np

# JUST FOR DEBUGGING ON VSCODE 
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

from self_supervised.TF2.models.networks.resnet50 import ResNet50
from self_supervised.TF2.utils.losses import mse_loss
from self_supervised.TF2.utils.accelerator import setup_accelerator

from self_supervised.TF2.models.simclr.simclr_flags import FLAGS
from absl import app


class EMA():
    """ Exponential moving average """
    def __init__(self, tau_base, total_steps):
        super().__init__()
        self.tau_base = tau_base
        self.total_steps = total_steps

        self.tau = tau_base

    def update_average(self, old, new, current_step):
        self.tau = 1 - (1-self.tau_base) * (tf.cos(np.pi * current_step / self.total_steps) + 1)/2
        if old is None:
            return new
        return old * self.tau + (1 - self.tau) * new


# class NetWrapper(tf.keras.Model):
#     """ Initializes the online network """

#     def __init__(self,
#         net: tf.keras.Model,
#         projection_size,
#         projection_hidden_size,
#         layer=hidden_layer):
#         super(NetWrapper, self).__init__()
#         self.net = net


# class MLP(tf.keras.Sequential):
#     """ MLP class for projector and predictor """
#     def __init__(self, in_shape, hidden_size, projection_size, name):
#         super(MLP, self).__init__()
#             # layers=[
#             # tf.keras.Input(shape=in_shape),
#             # tf.keras.layers.Flatten(),
#             # tf.keras.layers.Dense(hidden_size),  #, activation='linear'),  # TODO: check if equivalent to nn.Linear()
#             # tf.keras.layers.BatchNormalization(),
#             # tf.keras.layers.ReLU(),
#             # tf.keras.layers.Dense(projection_size)  #, activation='linear')
#             # ],
#             # name=name)

#         # self.layers = [
#         #     tf.keras.Input(shape=in_shape),
#         #     tf.keras.layers.Flatten(),
#         #     tf.keras.layers.Dense(hidden_size),  #, activation='linear'),  # TODO: check if equivalent to nn.Linear()
#         #     tf.keras.layers.BatchNormalization(),
#         #     tf.keras.layers.ReLU(),
#         #     tf.keras.layers.Dense(projection_size)  #, activation='linear')
#         # ]

#         self.add(tf.keras.Input(shape=in_shape))
#         self.add(tf.keras.layers.Flatten())
#         self.add(tf.keras.layers.Dense(hidden_size))
#         self.add(tf.keras.layers.BatchNormalization())
#         self.add(tf.keras.layers.ReLU())
#         self.add(tf.keras.layers.Dense(projection_size))

#         # self.net = tf.keras.Sequential(layers=[
#         #     tf.keras.Input(shape=in_shape),
#         #     tf.keras.layers.Dense(hidden_size),  #, activation='linear'),  # TODO: check if equivalent to nn.Linear()
#         #     tf.keras.layers.BatchNormalization(),
#         #     tf.keras.layers.ReLU(),
#         #     tf.keras.layers.Dense(projection_size)  #, activation='linear')
#         # ])

#     # def build(self, input_shape):
#     #     super(MLP, self).build(input_shape)

#     #     self.layers.build(input_shape)
#     #     self.built = True

#     def call(self, x, training=True):
#         # return self.net(x, training=training)

#         return super(MLP, self).call(x, training=training)


def MLP(name, hidden_size=1, projection_size=256, in_shape=None):
    #4096
    """ MLP head for projector and predictor """
    model = tf.keras.Sequential(name=name)

    if in_shape:
        model.add(tf.keras.Input(shape=in_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hidden_size))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(projection_size))

    return model


class BYOL(tf.keras.Model):

    def __init__(self, in_shape, backbone, tau_base=0.996, loss_fn=None, steps=80):
        super(BYOL, self).__init__()

        self.loss_fn = loss_fn
        self.steps = steps

        print(backbone.compute_output_shape((None, *in_shape)))

        self.online_network = tf.keras.Sequential([
            backbone,  # base encoder f, outputs feature space y as (8*img dims)
            tf.keras.layers.Flatten(),
            MLP(name="projection"),  # MLP projection g, maps feature space onto 
            MLP(name="predictor")
        ])  # NetWrapper()

        self.target_network = tf.keras.models.clone_model(self.online_network)
        # TODO: give EMA appr inputs
        self.target_ema = EMA(tau_base, steps)

    def build(self, input_shape):
        self.online_network.build(input_shape)
        self.built = True

    def call(self, x, training=False):

        result = self.online_network(x, training=training)
        return result

    def compile(self, optimizer, loss=mse_loss, **kwargs):
        super(BYOL, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss

    def summary(self, line_length=None, positions=None, print_fn=None):
        print("Online network:\n")
        self.online_network.summary()
        return super().summary(line_length=line_length, positions=positions, print_fn=print_fn)

    def shared_step(self, data, training):
        x, y = data

        last_channel = x.shape[-1]

        view_1 = x[..., :3]
        view_2 = x[..., 3:]

        online_network_out_1 = self.online_network(view_1, training)
        online_network_out_2 = self.online_network(view_2, training)

        target_network_out_1 = self.target_network(view_1, training)
        target_network_out_2 = self.target_network(view_2, training)

        loss = self.loss_fn(
            online_network_out_1,
            online_network_out_2,
            target_network_out_1,
            target_network_out_2
        )

        return loss

    @tf.function
    def train_step(self, data):
        # apply gradient tape to online network only
        """
        # https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
        """

        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        trainable_variables = self.online_network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        # self.update_moving_average()

        return {'train_loss': loss}

    @tf.function
    def test_step(self, data):
        pass

    def update_moving_average(self):
        assert self.target_network is not None, 'target encoder has not been created yet'

        for current_params, ma_params in zip(self.online_network.weights, self.target_network.weights):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def augment(self, data):
        # TODO: use imported augment utils
        return tf.layers.experimental.preprocessing.RandomFlip(data)


def main(argv):
    del argv

    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 'oliv')

    global_batch_size = FLAGS.num_cores * FLAGS.batch_size

    if FLAGS.dataset == 'cifar10':
        from self_supervised.TF2.dataset.cifar10 import load_input_fn
        import tensorflow_datasets as tfds

        train_ds = load_input_fn(split=tfds.Split.TRAIN,
                                 name='cifar10',
                                 batch_size=FLAGS.batch_size,
                                 training_mode='pretrain')

        val_ds = load_input_fn(split=tfds.Split.TEST,
                               name='cifar10',
                               batch_size=FLAGS.batch_size,
                               training_mode='pretrain')

        ds_info = tfds.builder(FLAGS.dataset).info
        steps_per_epoch = ds_info.splits['train'].num_examples // global_batch_size
        validation_steps = ds_info.splits['test'].num_examples // global_batch_size
        ds_shape = (32, 32, 3)

    with strategy.scope():

        if FLAGS.backbone == 'resnet50':

            backbone = ResNet50(include_top=False,
                                input_shape=ds_shape,
                                pooling=None)

        model = BYOL(
            in_shape=ds_shape,
            backbone=backbone
        )

        # model = MLP("test", in_shape=ds_shape)

        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(lr=FLAGS.learning_rate)

        # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=mse_loss, metrics=['accuracy'])
        model.build((None, *ds_shape))
        model.summary()

    # build model and compile it
    history = model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              batch_size=global_batch_size,
              epochs=10,
              validation_data=val_ds,
              validation_steps=validation_steps)

    print(history)


if __name__ == '__main__':
    app.run(main)
