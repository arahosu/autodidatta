import tensorflow as tf
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger

from absl import app, flags
import os
from datetime import datetime

from sss.datasets.oai_unlabelled import load_oai_full_dataset
from sss.network import projection_head, predictor_head
from sss.contrastive.simclr import SimCLR_UNet
from sss.contrastive.simsiam import SimSiam_UNet
from sss.utils import setup_accelerator
from sss.losses import nt_xent_loss

# Dataset
flags.DEFINE_integer(
    'image_size',
    288,
    'Image size to be used.')
flags.DEFINE_string(
    'dataset_dir',
    'dicom_files.txt',
    'set directory of your dataset')

# Training
flags.DEFINE_integer(
    'train_epochs', 200, 'Number of epochs to train the model')
flags.DEFINE_enum(
    'optimizer', 'adam', ['lamb', 'adam', 'sgd', 'adamw'],
    'optimizer for pre-training')
flags.DEFINE_float(
    'loss_temperature', 0.5, 'set temperature for loss function')
flags.DEFINE_integer('batch_size', 256, 'set batch size for pre-training.')
flags.DEFINE_float('learning_rate', 2e-04, 'set learning rate for optimizer.')
flags.DEFINE_integer(
    'hidden_dim', 512,
    'set number of units in the hidden \
     layers of the projection/predictor head')
flags.DEFINE_integer(
    'output_dim', 128,
    'set number of units in the output layer of the projection/predictor head')
flags.DEFINE_integer(
    'num_head_layers', 1,
    'set number of intermediate layers in the projection head')

# Model specification
flags.DEFINE_enum(
    'model',
    'simclr',
    ['simclr', 'simsiam', 'byol', 'barlow_twins'],
    'contrastive model to be trained')
flags.DEFINE_bool(
    'finetune_decoder_only',
    False,
    'whether to finetune decoder only during training'
)

# logging specification
flags.DEFINE_bool(
    'save_weights', False,
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
flags.DEFINE_string('tpu', 'oai-tpu', 'set the name of TPU device')
flags.DEFINE_bool('use_bfloat16', True, 'set whether to use mixed precision')

FLAGS = flags.FLAGS


def main(argv):

    del argv

    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)

    # load dataset:
    train_ds, val_ds = load_oai_full_dataset(
        FLAGS.dataset_dir,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        train_split=0.360,
        drop_remainder=True)

    steps_per_epoch = int(7786206 * 0.640) // FLAGS.batch_size
    validation_steps = int(7786206 * 0.360) // FLAGS.batch_size
    ds_shape = (FLAGS.image_size, FLAGS.image_size, 1)

    with strategy.scope():
        # load model
        classifier = None

        if FLAGS.model == 'simclr':
            model = SimCLR_UNet(
                input_shape=ds_shape,
                projection=projection_head(),
                classifier=classifier,
                finetune_decoder_only=FLAGS.finetune_decoder_only,
                loss_temperature=FLAGS.loss_temperature)
        elif FLAGS.model == 'simsiam':
            model = SimSiam_UNet(
                input_shape=ds_shape,
                projection=projection_head(
                    batch_norm_output=True),
                predictor=predictor_head(),
                classifier=classifier,
                finetune_decoder_only=FLAGS.finetune_decoder_only)

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

        model.compile(optimizer=optimizer,
                      loss_fn=loss_fn)

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
