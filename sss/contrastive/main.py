import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger

from absl import app
import os
from datetime import datetime

from sss.contrastive.flags import FLAGS
from sss.datasets.cifar10 import load_input_fn
from sss.datasets.oai import load_dataset
from sss.datasets.oai_unlabelled import load_oai_full_dataset
from sss.network import projection_head, predictor_head, projection_head_conv
from sss.resnet import ResNet18, ResNet34, ResNet50
from sss.vgg import VGG_UNet
from sss.contrastive.simclr import SimCLR, SimCLR_UNet
from sss.contrastive.simsiam import SimSiam, SimSiam_UNet
from sss.utils import setup_accelerator, LearningRateSchedule
from sss.losses import nt_xent_loss, tversky_loss
from sss.metrics import DiceMetrics, dice_coef_eval, dice_coef


def main(argv):

    del argv

    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)

    # load datasets:
    if FLAGS.dataset == 'cifar10':
        if FLAGS.model == 'supervised':
            training_mode = 'finetune'
        else:
            training_mode = 'pretrain'

        train_ds = load_input_fn(split='train',
                                 batch_size=FLAGS.batch_size,
                                 training_mode=training_mode,
                                 normalize=FLAGS.normalize)

        val_ds = load_input_fn(split='test',
                               batch_size=FLAGS.batch_size,
                               training_mode=training_mode,
                               normalize=FLAGS.normalize)

        ds_info = tfds.builder(FLAGS.dataset).info
        num_train_examples = ds_info.splits['train'].num_examples
        num_val_examples = ds_info.splits['test'].num_examples
        steps_per_epoch = num_train_examples // FLAGS.batch_size
        validation_steps = num_val_examples // FLAGS.batch_size
        ds_shape = (FLAGS.image_size, FLAGS.image_size, 3)

    elif FLAGS.dataset == 'oai':
        if FLAGS.model == 'supervised':
            training_mode = 'finetune'
        else:
            training_mode = 'pretrain'

        train_ds, val_ds = load_dataset(
            batch_size=FLAGS.batch_size,
            dataset_dir='gs://oai-challenge-dataset/tfrecords',
            training_mode=training_mode,
            fraction_data=FLAGS.fraction_data,
            multi_class=FLAGS.multi_class,
            add_background=FLAGS.add_background,
            normalize=FLAGS.normalize,
            buffer_size=int(19200 * FLAGS.fraction_data))

        steps_per_epoch = int(19200 * FLAGS.fraction_data) // FLAGS.batch_size
        validation_steps = 4480 // FLAGS.batch_size
        ds_shape = (FLAGS.image_size, FLAGS.image_size, 1)

        if not FLAGS.multi_class:
            activation = 'sigmoid'
            num_classes = 1
        else:
            if FLAGS.add_background:
                activation = 'softmax'
                num_classes = 7
            else:
                activation = 'sigmoid'
                num_classes = 6

    elif FLAGS.dataset == 'oai_full':
        train_ds, val_ds = load_oai_full_dataset(
            'dicom_files.txt',
            batch_size=FLAGS.batch_size,
            train_split=0.360,
            drop_remainder=True)

        steps_per_epoch = int(7786206 * 0.640) // FLAGS.batch_size
        validation_steps = int(7786206 * 0.360) // FLAGS.batch_size
        ds_shape = (288, 288, 1)

    with strategy.scope():
        # load model
        if FLAGS.dataset == 'cifar10':
            if FLAGS.backbone == 'resnet50':
                backbone = ResNet50(input_shape=ds_shape)
            elif FLAGS.backbone == 'resnet34':
                backbone = ResNet34(input_shape=ds_shape)
            elif FLAGS.backbone == 'resnet18':
                backbone = ResNet18(input_shape=ds_shape)
        elif FLAGS.dataset in ['oai', 'brats', 'oai_full']:
            backbone = VGG_UNet(input_shape=ds_shape)

        if FLAGS.online_ft:
            # load model for downstream task evaluation
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

            elif FLAGS.dataset in ['oai', 'brats', 'oai_full']:

                classifier = tfkl.Conv2D(
                    num_classes, (1, 1),
                    activation=activation, padding='same')

                loss = tversky_loss
                dice_metrics = [DiceMetrics(idx=idx) for idx in range(num_classes)]
                if FLAGS.multi_class:
                    if FLAGS.add_background:
                        dice_cartilage = dice_coef_eval
                    else:
                        dice_cartilage = dice_coef
                    metrics = [dice_metrics, dice_cartilage]
                else:
                    metrics = [dice_coef]
        else:
            classifier = None

        if FLAGS.model == 'simclr':
            if FLAGS.dataset == 'cifar10':
                model = SimCLR(
                    backbone=backbone,
                    projection=projection_head(),
                    classifier=classifier,
                    loss_temperature=FLAGS.loss_temperature)
            else:
                if FLAGS.pretrain_encoder_only:
                    model = SimCLR_UNet(
                        input_shape=ds_shape,
                        projection=projection_head(),
                        classifier=classifier,
                        finetune_decoder_only=FLAGS.finetune_decoder_only,
                        loss_temperature=FLAGS.loss_temperature)
                else:
                    model = SimCLR(
                        backbone=backbone,
                        projection=projection_head_conv(),
                        classifier=classifier,
                        loss_temperature=FLAGS.loss_temperature)
        elif FLAGS.model == 'simsiam':
            if FLAGS.dataset == 'cifar10':
                model = SimSiam(
                    backbone=backbone,
                    projection=projection_head(
                        batch_norm_output=True
                    ),
                    predictor=predictor_head(),
                    classifier=classifier
                )
            else:
                if FLAGS.pretrain_encoder_only:
                    model = SimSiam_UNet(
                        input_shape=ds_shape,
                        projection=projection_head(
                            batch_norm_output=True),
                        predictor=predictor_head(),
                        classifier=classifier,
                        finetune_decoder_only=FLAGS.finetune_decoder_only)
                else:
                    model = SimSiam(
                        backbone=backbone,
                        projection=projection_head_conv(
                            batch_norm_output=True,
                            flatten_output=False),
                        predictor=projection_head_conv())

        elif FLAGS.model == 'supervised':
            model = tf.keras.Sequential(
                [backbone,
                 classifier]
            )

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

        if FLAGS.model == 'simclr':
            loss_fn = nt_xent_loss
        elif FLAGS.model == 'simsiam':
            loss_fn = tf.keras.losses.cosine_similarity

        if classifier is not None:
            if FLAGS.model == 'supervised':
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )
            else:
                model.compile(
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    ft_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=FLAGS.ft_learning_rate),
                    loss=loss,
                    metrics=metrics)
        else:
            model.compile(optimizer=optimizer,
                          loss_fn=loss_fn)

        # model.build((None, *ds_shape))
        # model.summary()

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
        histfile = str(FLAGS.model) + '_results.csv'
        csv_logger = CSVLogger(os.path.join(histdir, histfile))
        cb = [csv_logger]

        # Save flag params in a flag file in the same subdirectory
        flagfile = os.path.join(histdir, 'train_flags.cfg')
        FLAGS.append_flags_into_file(flagfile)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.train_epochs,
        validation_data=val_ds,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=cb)

    if FLAGS.save_weights:
        weights_name = str(FLAGS.model) + '_weights.hdf5'
        model.save_weights(os.path.join(logdir, weights_name))


if __name__ == '__main__':
    app.run(main)
