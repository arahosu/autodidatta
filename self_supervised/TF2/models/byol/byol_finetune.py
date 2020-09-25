import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from absl import app

from self_supervised.TF2.models.simclr.simclr_flags import FLAGS
from self_supervised.TF2.utils.accelerator import setup_accelerator
from self_supervised.TF2.dataset.cifar10 import load_input_fn
from self_supervised.TF2.models.networks.resnet50 import ResNet50
from self_supervised.TF2.models.byol.byol_pretrain import BYOL

def main(argv):
    del argv
    # Set up accelerator
    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)
    # load datasets:
    if FLAGS.dataset == 'cifar10':
        train_ds = load_input_fn(split=tfds.Split.TRAIN,
                                 name='cifar10',
                                 batch_size=FLAGS.batch_size,
                                 training_mode='finetune',
                                 use_cloud=False if FLAGS.use_gpu else True,
                                 normalize=FLAGS.normalize)

        val_ds = load_input_fn(split=tfds.Split.TEST,
                               name='cifar10',
                               batch_size=FLAGS.batch_size,
                               training_mode='finetune',
                               use_cloud=False if FLAGS.use_gpu else True,
                               normalize=FLAGS.normalize)

        ds_info = tfds.builder(FLAGS.dataset).info
        steps_per_epoch = ds_info.splits['train'].num_examples // FLAGS.batch_size
        validation_steps = ds_info.splits['test'].num_examples // FLAGS.batch_size
        ds_shape = (32, 32, 3)

    with strategy.scope():

        byol_model = BYOL(in_shape=ds_shape,
                          hidden_size=512,
                          projection_size=128)
        model = byol_model.online_network
        model.load_weights(FLAGS.weights).expect_partial()
        model.trainable = False  # Freeze the weights

        if FLAGS.eval_linear:
            evaluator = tf.keras.Sequential([
                model.backbone,
                tfkl.Flatten(),
                tfkl.Dropout(rate=0.2),
                tfkl.Dense(10, activation='softmax')
            ])
        else:
            evaluator = tf.keras.Sequential([
                model.backbone,
                tfkl.Flatten(),
                tfkl.Dropout(rate=0.2),
                tfkl.Dense(512, use_bias=False),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Dropout(rate=0.2),
                tfkl.Dense(10, activation='softmax')
            ])

        optimizer = tfa.optimizers.AdamW(weight_decay=1e-06, learning_rate=1e-03)

        evaluator.compile(optimizer=optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
        evaluator.build((None, *ds_shape))
        evaluator.summary()

    evaluator.fit(train_ds,
                  steps_per_epoch=steps_per_epoch,
                  batch_size=FLAGS.batch_size,
                  epochs=FLAGS.eval_epochs,
                  validation_data=val_ds,
                  validation_steps=validation_steps)

if __name__ == '__main__':
    app.run(main)