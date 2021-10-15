from absl import app
from absl import flags
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger

import autodidatta.augment as A
from autodidatta.datasets import cifar10, stl10
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.mlp import projection_head, predictor_head
from autodidatta.utils.accelerator import setup_accelerator


# Dataset and Augmentation
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'stl10', 'imagenet'],
    'cifar10 (default), stl10, imagenet')
flags.DEFINE_integer(
    'image_size', 32,
    'image size to be used')
flags.DEFINE_float(
    'brightness', 0.4,
    'random brightness factor')
flags.DEFINE_float(
    'contrast', 0.4,
    'random contrast factor')
flags.DEFINE_float(
    'saturation', 0.4,
    'random saturation factor')
flags.DEFINE_float(
    'hue', 0.1,
    'random hue factor')
flags.DEFINE_list(
    'prob_solarization', [0.0, 0.0],
    'probability of applying solarization augmentation')

# Training
flags.DEFINE_integer(
    'train_epochs', 1000, 'Number of epochs to train the model')
flags.DEFINE_enum(
    'optimizer', 'adam', ['lamb', 'adam', 'sgd', 'adamw'],
    'optimizer for pre-training')
flags.DEFINE_integer('batch_size', 512, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 1e-03, 'set learning rate for optimizer.')
flags.DEFINE_integer(
    'hidden_dim', 2048,
    'set number of units in the hidden \
     layers of the projection/predictor head')
flags.DEFINE_integer(
    'output_dim', 512,
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
    ['resnet50', 'resnet34', 'resnet18', 'vgg_unet'],
    'resnet50 (default), resnet18, resnet34, vgg_unet')

# logging specification
flags.DEFINE_bool(
    'save_weights', True,
    'Whether to save weights. If True, weights are saved in logdir')
flags.DEFINE_bool(
    'save_history', True,
    'Whether to save the training history.'
)
flags.DEFINE_string(
    'histdir', '/home/User/Self-Supervised-Segmentation/training_logs',
    'Directory for where the training history is being saved'
)
flags.DEFINE_string(
    'logdir', '/home/User/Self-Supervised-Segmentation/weights',
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


class SimSiam(tf.keras.Model):

    def __init__(self,
                 backbone,
                 projection,
                 predictor,
                 classifier=None):

        super(SimSiam, self).__init__()

        self.backbone = backbone
        self.projection = projection
        self.predictor = predictor
        self.classifier = classifier

    def build(self, input_shape):

        self.backbone.build(input_shape)
        self.projection.build(self.backbone.compute_output_shape(input_shape))
        self.predictor.build(
            self.projection.compute_output_shape(
                self.backbone.compute_output_shape(input_shape)))

        if self.classifier is not None:
            self.classifier.build(
                self.backbone.compute_output_shape(input_shape))

        self.built = True

    def call(self, x, training=False):

        result = self.backbone(x, training=training)

        return result

    def compile(self, loss_fn, ft_optimizer=None, **kwargs):
        super(SimSiam, self).compile(**kwargs)
        self.loss_fn = loss_fn
        if self.classifier is not None:
            assert ft_optimizer is not None, \
                'ft_optimizer should not be None if self.classifier is not \
                    None'
            self.ft_optimizer = ft_optimizer

    def compute_output_shape(self, input_shape):

        current_shape = self.backbone.compute_output_shape(input_shape)

        return current_shape

    def shared_step(self, data, training):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        feat_i = self.backbone(xi, training=training)
        feat_j = self.backbone(xj, training=training)

        if isinstance(feat_i, list):
            feat_i = feat_i[-1]
            feat_j = feat_j[-1]

        zi = self.projection(feat_i, training=training)
        zj = self.projection(feat_j, training=training)

        pi = self.predictor(zi, training=training)
        pj = self.predictor(zj, training=training)

        loss = self.loss_fn(pi, tf.stop_gradient(zj)) / 2
        loss += self.loss_fn(pj, tf.stop_gradient(zi)) / 2

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
            if isinstance(features, list):
                features = features[-1]
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
        trainable_variables = self.backbone.trainable_variables + \
            self.projection.trainable_variables + \
            self.predictor.trainable_variables
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
            features = self.backbone(view, training=False)
            if isinstance(features, list):
                features = features[-1]
            y_pred = self.classifier(features, training=False)
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'similarity_loss': loss, **metric_results}
        else:
            return {'similarity_loss': loss}

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None,
                     save_backbone_only=False):
        if save_backbone_only:
            weights = self.backbone.save_weights(
                filepath, overwrite, save_format, options)
        else:
            weights = super(SimSiam, self).save_weights(
                filepath, overwrite, save_format, options)
        return weights


def main(argv):

    del argv

    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)

    # Define augmentation functions
    aug_fn_1 = A.Augment([
        A.layers.RandomResizedCrop(
            FLAGS.image_size, FLAGS.image_size),
        A.layers.ColorJitter(
            FLAGS.brightness,
            FLAGS.contrast, 
            FLAGS.saturation,
            FLAGS.hue, p=0.8),
        A.layers.ToGray(p=0.2),
        A.layers.Solarize(p=FLAGS.prob_solarization[0])
        ])
        
    aug_fn_2 = A.Augment([
        A.layers.RandomResizedCrop(
            FLAGS.image_size, FLAGS.image_size),
        A.layers.ColorJitter(
            FLAGS.brightness,
            FLAGS.contrast, 
            FLAGS.saturation,
            FLAGS.hue, p=0.8),
        A.layers.ToGray(p=0.2),
        A.layers.Solarize(p=FLAGS.prob_solarization[1])
        ])

    # Define image dimension
    ds_shape = (FLAGS.image_size, FLAGS.image_size, 3)

    if FLAGS.dataset == 'cifar10':
        load_dataset = cifar10.load_input_fn
    elif FLAGS.dataset == 'stl10':
        load_dataset = stl10.load_input_fn

    train_ds = load_dataset(
        is_training=True,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        pre_train=True,
        aug_fn=aug_fn_1,
        aug_fn_2=aug_fn_2)
    validation_ds = load_dataset(
        is_training=False,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        pre_train=True,
        aug_fn=aug_fn_1,
        aug_fn_2=aug_fn_2)

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
            assert FLAGS.dataset != 'stl10', \
                'Online finetuning is not supported for stl10'

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

        model = SimSiam(backbone=backbone,
                        projection=projection_head(
                            hidden_dim=FLAGS.hidden_dim,
                            output_dim=FLAGS.output_dim,
                            num_layers=FLAGS.num_head_layers,
                            batch_norm_output=True),
                        predictor=predictor_head(
                            hidden_dim=FLAGS.hidden_dim,
                            output_dim=FLAGS.output_dim,
                            num_layers=FLAGS.num_head_layers),
                        classifier=classifier)

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
                loss_fn=tf.keras.losses.cosine_similarity,
                ft_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=FLAGS.ft_learning_rate),
                loss=loss,
                metrics=metrics)
        else:
            model.compile(optimizer=optimizer,
                          loss_fn=tf.keras.losses.cosine_similarity)

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
        histfile = 'simsiam_results.csv'
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
        weights_name = 'simsiam_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name),
                           save_backbone_only=True)


if __name__ == '__main__':
    app.run(main)
