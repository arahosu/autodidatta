import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam, SGD

from absl import app
from sss.flags import FLAGS

from sss.cifar10 import load_input_fn
from sss.oai import load_dataset
from sss.network import projection_head, predictor_head
from sss.resnet import ResNet18, ResNet34, ResNet50
from sss.vgg import VGG_UNet, VGG16, SegNet_Decoder
from sss.contrastive import SimCLR, SimSiam
from sss.utils import setup_accelerator, LearningRateSchedule
from sss.losses import tversky_loss
from sss.metrics import DiceMetrics, dice_coef_eval


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
                                 training_mode='finetune',
                                 normalize=FLAGS.normalize)

        val_ds = load_input_fn(split='test',
                               batch_size=FLAGS.batch_size,
                               training_mode='finetune',
                               normalize=FLAGS.normalize)

        ds_info = tfds.builder(FLAGS.dataset).info
        num_train_examples = ds_info.splits['train'].num_examples
        num_val_examples = ds_info.splits['test'].num_examples
        steps_per_epoch = num_train_examples // FLAGS.batch_size
        validation_steps = num_val_examples // FLAGS.batch_size
        ds_shape = (FLAGS.image_size, FLAGS.image_size, 3)

    elif FLAGS.dataset == 'oai':

        train_ds, val_ds = load_dataset(
            batch_size=FLAGS.batch_size,
            dataset_dir='gs://oai-challenge-dataset/tfrecords',
            training_mode='finetune',
            fraction_data=FLAGS.fraction_data,
            buffer_size=int(19200 * FLAGS.fraction_data))

        steps_per_epoch = int(19200 * FLAGS.fraction_data) // FLAGS.batch_size
        validation_steps = 4480 // FLAGS.batch_size
        ds_shape = (FLAGS.image_size, FLAGS.image_size, 1)

        if FLAGS.output_activation == 'softmax':
            num_classes = 7
        elif FLAGS.output_activation == 'sigmoid':
            num_classes = 6

    with strategy.scope():
        # initialise backbone
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)
        elif FLAGS.backbone == 'vgg16':
            backbone = VGG16(input_shape=ds_shape)
        elif FLAGS.backbone == 'vgg_unet':
            backbone = VGG_UNet(input_shape=ds_shape)

        # load model
        if FLAGS.model == 'simclr':
            pretrain_model = SimCLR(
                backbone=backbone,
                projection=projection_head(),
                classifier=None,
                loss_temperature=0.5)

        elif FLAGS.model == 'simsiam':
            pretrain_model = SimSiam(
                backbone=backbone,
                projection=projection_head(
                    batch_norm_output=True
                ),
                predictor=predictor_head(),
                classifier=None)

        pretrain_model.build((None, *ds_shape))
        pretrain_model.load_weights(FLAGS.weights)

        # initialise new model for downstream task evaluation
        if FLAGS.dataset == 'cifar10':
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

        elif FLAGS.dataset in ['oai', 'brats']:

            if FLAGS.backbone == 'vgg_unet':
                classifier = tfkl.Conv2D(
                    num_classes, (1, 1),
                    activation=FLAGS.output_activation, padding='same')
            else:
                if FLAGS.seg_classifier == 'fcn':
                    classifier = tf.keras.Sequential(
                        [tfkl.Conv2D(
                            num_classes, (1, 1),
                            activation=FLAGS.output_activation,
                            padding='valid'),
                         tfkl.UpSampling2D(
                            (32, 32), interpolation='bilinear')]
                    )
                elif FLAGS.seg_classifier == 'segnet':
                    classifier = tf.keras.Sequential(
                        [SegNet_Decoder(),
                         tfkl.Conv2D(
                             num_classes, (1, 1),
                             activation=FLAGS.output_activation,
                             padding='same')]
                    )

            loss = tversky_loss
            dice_metrics = [DiceMetrics(idx=idx) for idx in range(num_classes)]
            metrics = [dice_metrics, dice_coef_eval]

        model = tf.keras.Sequential(
            [pretrain_model.backbone,
             classifier])

        if FLAGS.custom_schedule:
            lr_rate = LearningRateSchedule(steps_per_epoch,
                                           FLAGS.learning_rate,
                                           1e-09,
                                           0.8,
                                           list(range(1, FLAGS.train_epochs)),
                                           0)
        else:
            lr_rate = FLAGS.learning_rate

        if FLAGS.optimizer == 'lamb':
            optimizer = LAMB(
                learning_rate=lr_rate,
                weight_decay_rate=1e-04,
                exclude_from_weight_decay=['bias', 'BatchNormalization'])
        elif FLAGS.optimizer == 'adam':
            optimizer = Adam(learning_rate=lr_rate)
        elif FLAGS.optimizer == 'sgd':
            optimizer = SGD(learning_rate=lr_rate)
        elif FLAGS.optimizer == 'adamw':
            optimizer = AdamW(
                weight_decay=1e-06, learning_rate=lr_rate)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)

        model.build((None, *ds_shape))
        model.fit(train_ds,
                  steps_per_epoch=steps_per_epoch,
                  epochs=FLAGS.train_epochs,
                  validation_data=val_ds,
                  validation_steps=validation_steps,
                  verbose=1)


if __name__ == '__main__':
    app.run(main)
