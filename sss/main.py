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
from sss.utils import setup_accelerator
from sss.losses import nt_xent_loss, tversky_loss
from sss.contrastive import SimCLR, SimSiam


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
                                 training_mode='pretrain',
                                 normalize=FLAGS.normalize)

        val_ds = load_input_fn(split='test',
                               batch_size=FLAGS.batch_size,
                               training_mode='pretrain',
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
            training_mode='pretrain')

        steps_per_epoch = 19200 // FLAGS.batch_size
        validation_steps = 4480 // FLAGS.batch_size
        ds_shape = (FLAGS.image_size, FLAGS.image_size, 1)

    with strategy.scope():
        # load model
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)

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

            if FLAGS.model == 'simclr':
                model = SimCLR(
                    backbone=backbone,
                    projection=projection_head(),
                    classifier=classifier,
                    loss_temperature=0.5)
            elif FLAGS.model == 'simsiam':
                model = SimSiam(
                    backbone=backbone,
                    projection=projection_head(
                        hidden_dim=2048,
                        output_dim=2048,
                        num_layers=1,
                        batch_norm_output=True
                    ),
                    predictor=predictor_head(
                        hidden_dim=2048,
                        output_dim=2048),
                    classifier=classifier
                )

            loss = tf.keras.losses.sparse_categorical_crossentropy
            metrics = ['acc']

        # elif FLAGS.dataset in ['oai', 'brats']:
        #     classifier = VGG_UNet_Decoder(
        #         6, output_activation='sigmoid',
        #         use_transpose=False)

        #     model = SimCLR_UNet(backbone=backbone,
        #                         projection=projection_head(),
        #                         classifier=classifier,
        #                         loss_temperature=0.5)

        #     loss = tversky_loss
        #     dice_metrics = [DiceMetrics(idx=idx) for idx in range(6)]
        #     # metrics = [dice_metrics, dice_coef_eval]
        #     metrics = [dice_metrics]

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
            if FLAGS.model == 'simclr':
                loss_fn = nt_xent_loss
            elif FLAGS.model == 'simsiam':
                loss_fn = tf.keras.losses.cosine_similarity

            model.compile(
                optimizer=optimizer,
                loss_fn=loss_fn,
                ft_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=FLAGS.ft_learning_rate),
                loss=loss,
                metrics=metrics)
        else:
            model.compile(optimizer=optimizer,
                          loss_fn=nt_xent_loss)

        model.build((None, *ds_shape))
        model.backbone.summary()

    # Define checkpoints
    # time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = os.path.join(FLAGS.logdir, time)

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=FLAGS.train_epochs,
              validation_data=val_ds,
              validation_steps=validation_steps,
              verbose=1)

    # model.save_weights(os.path.join(logdir, 'simclr_weights.hdf5'))


if __name__ == '__main__':
    app.run(main)
