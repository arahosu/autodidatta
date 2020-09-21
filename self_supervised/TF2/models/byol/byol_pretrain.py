# python3 -m Self-Supervised-Segmentation.self_supervised.TF2.models.byol.byol_pretrain

import tensorflow as tf
# import tensorflow.keras.layers as tfkl
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.optimizers import Adam
import numpy as np
from itertools import cycle

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
        #TODO: make sure total steps are equal to epoch steps
        self.total_steps = total_steps

        self.tau = tau_base

    # def update_average2(self, old, new, current_step):
    #     self.tau = 1 - ((1-self.tau_base) * (tf.math.cos(np.pi * current_step / self.total_steps) + 1)/2)

    #     def ma_update_fn(old, new):
    #         return tf.math.scalar_mul(self.tau, old) + tf.math.scalar_mul((1 - self.tau), new)

    #     if old is None:
    #         return new
    #     return [ma_update_fn(old_i, new_i) for old_i, new_i in zip(old, new)]

    def update_average(self, old, new, current_step):
        # Must be numpy calculation else use: self.tau = 1 - ((1-self.tau_base) * (tf.math.cos(np.pi * current_step / self.total_steps) + 1)/2)
        self.tau = 1 - ((1-self.tau_base) * (np.cos(np.pi * current_step / self.total_steps) + 1)/2)

        def ma_update_fn(old, new):
            if not old:
                return new
            if isinstance(old, list):
                old_all = []
                for old_weights, new_weights in zip(old, new):
                    old_all.append((self.tau * old_weights) + ((1 - self.tau) * new_weights))
                return old_all
            return tf.math.scalar_mul(self.tau, old) + tf.math.scalar_mul((1 - self.tau), new)

        if not old.trainable_variables:
            return new

        updated_weights = []
        for old_i, new_i in zip(old.layers, new.layers):
            if isinstance(new_i, tf.keras.layers.BatchNormalization) or not new_i or not new_i.trainable:
                # updated_weights.append(new_i.get_weights())
                updated_weights.append(new_i)
            else:
                old_i.set_weights(ma_update_fn(old_i.get_weights(), new_i.get_weights()))
                updated_weights.append(old_i)
 
        return updated_weights
        # Same as above
        # return [ma_update_fn(old_i.get_weights(), new_i.get_weights()) for old_i, new_i in zip(old.layers, new.layers)]

# class NetWrapper(tf.keras.Model):
#     """ Initializes the online network """

#     def __init__(self,
#         net: tf.keras.Model,
#         projection_size,
#         projection_hidden_size,
#         layer=hidden_layer):
#         super(NetWrapper, self).__init__()
#         self.net = net


def MLP(name, hidden_size=4096, projection_size=256, in_shape=None):
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

    def __init__(self, in_shape, backbone='resnet50', tau_base=0.996, loss_fn=None, steps=80):
        super(BYOL, self).__init__()

        # For EMA updater
        # TODO: delete loss fn? 
        self.loss_fn = loss_fn
        self.steps = steps
        self.current_step = 0

        # Model
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(include_top=False,
                                input_shape=in_shape,
                                pooling=None)

        print(backbone.compute_output_shape((None, *in_shape)))

        self.online_network = tf.keras.Sequential([
            backbone,  # base encoder f, outputs feature space y as (8*img dims)
            tf.keras.layers.Flatten(),
            MLP(name="projection"),  # MLP projection g, maps feature space onto 
            MLP(name="predictor")
        ], name="online_network")  # NetWrapper()

        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(include_top=False,
                                input_shape=in_shape,
                                pooling=None)

        self.target_network = tf.keras.Sequential([
            backbone,  # base encoder f, outputs feature space y as (8*img dims)
            tf.keras.layers.Flatten(),
            MLP(name="projection"),  # MLP projection g, maps feature space onto 
            MLP(name="predictor_")
        ], name="target_network")

        # TODO: give EMA appr inputs for step
        self.target_ema_updater = EMA(tau_base, steps)

        # TODO: delete
        self.update_moving_average()

    def augment(self, data):
        # TODO: use imported augment utils
        return tf.layers.experimental.preprocessing.RandomFlip(data)

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
        self.online_network.summary()
        # return super().summary(line_length=line_length, positions=positions, print_fn=print_fn)

    def shared_step(self, data, training):
        x, y = data

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
        self.current_step += 1

        with tf.GradientTape() as tape:
            loss = self.shared_step(data, training=True)
        trainable_variables = self.online_network.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        self.update_moving_average()

        return {'train_loss': loss}

    @tf.function
    def test_step(self, data):
        pass

    # @tf.function
    def update_moving_average(self):
        assert self.target_network is not None, 'target encoder has not been created yet'
        
        # updated_target_weights = (self.target_ema_updater.update_average(
        #     self.target_network.trainable_variables,
        #     self.online_network.trainable_variables,
        #     self.current_step
        # ))

        # Unfortunately only set_weights can modify tranable varuables 

        for i in range(len(self.target_network.layers)):
            self.target_network.layers[i].set_weights(self.target_ema_updater.update_average(
                self.target_network.layers[i],
                self.online_network.layers[i],
                self.current_step
            ))

        # for i in range(len(self.target_network.trainable_variables)):
        #     self.target_network.trainable_variables.__setitem__(i, updated_target_weights[i])

        # Get names of each var in trainable_variables, 
        # match to the layer.[i].kernel.name for index of names
        # trainable_layer_names = [var.name for var in self.target_network.trainable_variables]

        # name_cyc = enumerate(trainable_layer_names)
        # i, name = next(name_cyc)
        # for block_i in range(len(self.target_network.layers)):
        #     if not self.target_network.layers[block_i].trainable_variables:
        #         # If this layer or layer block has no trainable vars continue
        #         continue

        #     for layer_i in range(len(self.target_network.layers[block_i].layers)):
        #         print(self.target_network.layers[block_i].layers[layer_i].name)
                
        #         if self.target_network.layers[block_i].layers[layer_i].name == name:
        #             self.target_network.layers[block_i].layers[layer_i].set_weights(updated_target_weights[i])
        #             i, name = next(name_cyc)


        # for i, name in enumerate(trainable_layer_names):
        #     # Check that current layer names match

        #     # if self.target_network.layers[block_i]._object_identifier == '_tf_keras_layer':
        #     #     current_layer = self.target_network.layers[block_i]
        #     #     block_i += 1
        #     # else:
        #     #     current_layer = self.target_network.layers[block_i].layer[layer_i]

        #     if not self.target_network.layers[block_i].trainable_variables:
        #         continue

        #     for layer_i in range(len(self.target_network.layers[block_i].layers)):
        #         if self.target_network.layers[block_i].layers[layer_i].name == name:
        #             self.target_network.layers[block_i].layers[layer_i].set_weights(updated_target_weights[i])
            
        #     block_i += 1
        # for i in range(len(self.target_network.layers)):
        #     self.target_network.layer[i].set_weights(self.target_ema_updater.update_average(
        #         self.online_network.trainable_variables,
        #         self.target_network.trainable_variables,
        #         self.current_step
        #     )

        # for current_params, ma_params in zip(self.online_network.trainable_variables, self.target_network.trainable_variables):
        #     old_weight, up_weight = ma_params.data, current_params.data
        #     ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight, self.current_step)


def main(argv):
    del argv

    # Set up accelerator
    # strategy = setup_accelerator(FLAGS.use_gpu,
    #                              FLAGS.num_cores,
    #                              'oliv')

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

    # with strategy.scope():

    model = BYOL(
        in_shape=ds_shape,
        backbone=FLAGS.backbone
    )

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
