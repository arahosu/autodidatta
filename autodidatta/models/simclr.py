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
from autodidatta.datasets import Dataset
from autodidatta.flags import dataset_flags, training_flags, utils_flags
from autodidatta.models.base import BaseModel
from autodidatta.models.networks.mlp import projection_head
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.utils.accelerator import setup_accelerator
from autodidatta.utils.loss import nt_xent_loss, nt_xent_loss_v2

# Redefine default value
flags.FLAGS.set_default('output_dim', 256)
FLAGS = flags.FLAGS

class SimCLR(BaseModel):

    def __init__(self,
                 backbone,
                 projector,
                 classifier=None,
                 loss_temperature=0.5):

        super(SimCLR, self).__init__(
            backbone=backbone,
            projector=projector,
            predictor=None,
            classifier=classifier
        )

        self.loss_temperature = loss_temperature

    def shared_step(self, data, training):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        zi = self.backbone(xi, training=training)
        zj = self.backbone(xj, training=training)

        if self.projector is not None:
            zi = self.projector(zi, training=training)
            zj = self.projector(zj, training=training)

        zi = tf.math.l2_normalize(zi, axis=-1)
        zj = tf.math.l2_normalize(zj, axis=-1)

        loss = self.loss_fn(
            hidden1=zi, hidden2=zj,
            temperature=self.loss_temperature,
            strategy=self.distribute_strategy)

        return loss


def main(argv):

    del argv

    # Choose accelerator 
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)
    
    # Choose whether to train with float32 or bfloat16 precision
    if FLAGS.use_bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Select dataset
    if FLAGS.dataset == 'cifar10':
        image_size = 32
        train_split = 'train'
        validation_split = 'test'
    elif FLAGS.dataset == 'stl10':
        image_size = 96
        train_split = 'train' if not online_ft else 'unlabelled'
        validation_split = 'test'
    elif FLAGS.dataset == 'imagenet2012':
        assert FLAGS.dataset_dir is not None, 'for imagenet2012, \
            dataset direcotry must be specified'
        image_size = 224
        train_split = 'train'
        validation_split = 'validation'
    else:
        raise NotImplementedError("other datasets have not yet been implmented")

    # Define augmentation functions
    augment_kwargs = dataset_flags.parse_augmentation_flags()

    aug_fn_1 = A.SSLAugment(
        image_size=image_size,
        gaussian_prob=FLAGS.gaussian_prob[0],
        solarization_prob=FLAGS.solarization_prob[0],
        **augment_kwargs)
    aug_fn_2 = A.SSLAugment(
        image_size=image_size,
        gaussian_prob=FLAGS.gaussian_prob[1],
        solarization_prob=FLAGS.solarization_prob[1],
        **augment_kwargs)

    # Define dataloaders
    train_loader = Dataset(
        FLAGS.dataset,
        train_split,
        FLAGS.dataset_dir,
        aug_fn_1, aug_fn_2)
    validation_loader = Dataset(
        FLAGS.dataset,
        validation_split,
        FLAGS.dataset_dir,
        aug_fn_1, aug_fn_2)

    # Define datasets from the dataloaders
    train_ds = train_loader.load(
        FLAGS.batch_size,
        image_size,
        True,
        True,
        use_bfloat16=FLAGS.use_bfloat16)

    validation_ds = validation_loader.load(
        FLAGS.batch_size,
        image_size,
        False,
        True,
        use_bfloat16=FLAGS.use_bfloat16)
    
    # Get number of examples from dataloaders
    num_train_examples = train_loader.dataset_size
    num_val_examples = validation_loader.dataset_size
    steps_per_epoch = num_train_examples // FLAGS.batch_size
    validation_steps = num_val_examples // FLAGS.batch_size
    ds_shape = (image_size, image_size, 3)

    with strategy.scope():
        # Define backbone
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)
        else:
            raise NotImplementedError("other backbones have not yet been implemented")

        # If online finetuning is enabled
        if FLAGS.online_ft:
            assert FLAGS.dataset != 'stl10', \
                'Online finetuning is not supported for stl10'

            # load classifier for downstream task evaluation
            classifier = training_flags.load_classifier()

            finetune_loss = tf.keras.losses.sparse_categorical_crossentropy
            metrics = ['acc']
        else:
            classifier = None

        model = SimCLR(backbone=backbone,
                       projector=projection_head(
                           hidden_dim=FLAGS.proj_hidden_dim,
                           output_dim=FLAGS.output_dim,
                           num_layers=FLAGS.num_head_layers,
                           batch_norm_output=True),
                       classifier=classifier,
                       loss_temperature=FLAGS.loss_temperature)

        # load_optimizer
        optimizer, ft_optimizer = training_flags.load_optimizer(num_train_examples)

        if FLAGS.online_ft:
            model.compile(
                optimizer=optimizer,
                loss_fn=nt_xent_loss_v2,
                ft_optimizer=ft_optimizer,
                loss=finetune_loss,
                metrics=metrics)
        else:
            model.compile(optimizer=optimizer,
                          loss_fn=nt_xent_loss_v2)

        # Build the model
        model.build((None, *ds_shape))
        model.summary()

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
        histfile = 'simclr_results.csv'
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
        weights_name = 'simclr_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name), save_backbone_only=True)


if __name__ == '__main__':
    app.run(main)
