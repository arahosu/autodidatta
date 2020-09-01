import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.regularizers import l1
from tensorflow.keras.applications.resnet50 import ResNet50

from self_supervised.TF2.utils.losses import nt_xent_loss

def get_projection_head(use_2D=True,
                        use_batchnorm=True,
                        activation='relu',
                        num_layers=1,
                        proj_head_dim=2048,
                        proj_head_reg=1e-06,
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

        self.backbone = backbone
        if projection is None:
            self.projection = get_projection_head()
        else:
            self.projection = projection
        self.loss_temperature = loss_temperature

    def call(self, x, training=False):

        result = self.backbone(x, training=training)

        return result

    def compile(self, optimizer, loss_fn=nt_xent_loss):
        super(SimCLR, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def shared_step(self, data, training):

        xi, xj = data

        hi = self.backbone(xi, training=training)
        hj = self.backbone(xj, trainnig=training)

        zi = self.projection(hi, training=training)
        zj = self.projection(hj, training=training)

        zi = tf.math.l2_normalize(zi, axis=1)
        zj = tf.math.l2_normalize(zj, axis=1)

        loss = self.loss_fn(zi, zj, self.loss_temperature)

        return loss

    def train_step(self, data):

        with tf.GradientTape as tape:
            loss = self.shared_step(data, training=True)
        trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        return {'train_loss': loss}

    def test_step(self, data):

        loss = self.shared_step(data, training=False)

        return {'validation_loss': loss}

if __name__ == '__main__':

    """ Get Flags """
    from absl import flags

    # Define flags for pre-training

    # Dataset
    flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'BraTS', 'OAI'], 'cifar10 (default), BraTS, OAI')
    flags.DEFINE_string('dataset_dir', '.', 'set directory of your dataset')

    # Training
    flags.DEFINE_bool('online_finetune', True, 'set whether to run online finetuner')
    flags.DEFINE_enum('optimizer', 'lars', ['lars', 'adam'], 'lars (default), adam')
    flags.DEFINE_int('batch_size', '512', 'set batch size for pre-training.')
    flags.DEFINE_float('learning_rate', '1.', 'set learning rate for optimizer.')
    flags.DEFINE_float('lars_momentum', '0.9', 'set momentum for lars optimizer.')
    flags.DEFINE_int('lars_sched_step', '30', 'set schedule step for lars optimizer')
    flags.DEFINE_float('lar_gamma', '0.5', 'set gamma for lars optimizer')
    flags.DEFINE_flags('weight_decay', '1e-04', 'set weight decay')

    # Model specification
    flags.DEFINE_enum('backbone', 'resnet50', ['resnet50', 'vgg16', 'vgg19'], 'resnet50 (default)')
    flags.DEFINE_bool('use_2D', True, 'set whether to train on 2D or 3D data. Required for BraTS and OAI only')
    flags.DEFINE_float('loss temperature', '0.5', 'set temperature for loss function')
    flags.DEFINE_bool('use_gpu', 'False', 'set whether to use GPU')
    flags.DEFINE_int('num_cores', '8', 'set number of cores/workers for TPUs/GPUs')
    flags.DEFINE_str('tpu', 'oai-tpu', 'set the name of TPU device')

    FLAGS = flags.FLAGS
