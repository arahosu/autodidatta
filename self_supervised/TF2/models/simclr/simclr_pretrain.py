import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1

from self_supervised.TF2.models.networks.resnet50 import ResNet50
from self_supervised.TF2.utils.accelerator import setup_accelerator
from self_supervised.TF2.utils.losses import nt_xent_loss
from self_supervised.TF2.utils.lr_finder import LRFinder
from self_supervised.TF2.dataset.cifar10 import load_input_fn

from self_supervised.TF2.models.simclr.simclr_flags import FLAGS
from absl import app

def get_projection_head(use_2D=True,
                        use_batchnorm=True,
                        activation='relu',
                        num_layers=1,
                        proj_head_dim=2048,
                        proj_head_reg=1e-04,
                        output_dim=128
                        ):

    model = tf.keras.Sequential()

    if use_2D:
        model.add(tfkl.GlobalAveragePooling2D())
    else:
        model.add(tfkl.GlobalAveragePooling3D())

    model.add(tfkl.Flatten())

    for _ in range(num_layers):
        model.add(tfkl.Dense(proj_head_dim, kernel_regularizer=l1(proj_head_reg)))
        if use_batchnorm:
            model.add(tfkl.BatchNormalization())
        model.add(tfkl.Activation(activation))

    model.add(tfkl.Dense(proj_head_dim, kernel_regularizer=l1(proj_head_reg)))

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

        zi = tf.math.l2_normalize(zi, axis=1)
        zj = tf.math.l2_normalize(zj, axis=1)

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

        return {'validation_loss': loss}

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
                                 training_mode='pretrain')

        # strategy.experimental_distribute_dataset(train_ds)

        val_ds = load_input_fn(split=tfds.Split.TEST,
                               name='cifar10',
                               batch_size=FLAGS.batch_size,
                               training_mode='pretrain')

        # strategy.experimental_distribute_dataset(val_ds)

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
                       projection=get_projection_head(proj_head_reg=FLAGS.weight_decay),
                       loss_temperature=FLAGS.loss_temperature)

        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(lr=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'sgd':
            optimizer = SGD(lr=FLAGS.learning_rate, momentum=0.9)

        model.compile(optimizer=optimizer, loss_fn=nt_xent_loss)
        model.build((None, *ds_shape))
        model.summary()

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              batch_size=global_batch_size,
              epochs=100,
              validation_data=val_ds,
              validation_steps=validation_steps,
              verbose=1)
    
if __name__ == '__main__':
    app.run(main)
