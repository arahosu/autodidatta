import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

from absl import app
import os
from datetime import datetime

from sss.utils import setup_accelerator
from sss.datasets.oai import load_dataset, parse_fn_restore
from sss.metrics import DiceMetrics, dice_coef_eval, dice_coef
from sss.losses import tversky_loss
from sss.utils import LearningRateSchedule
from sss.baseline.flags import FLAGS
from sss.baseline.rotation import RotationPrediction
from sss.augmentation.base import sample_permutations


class Jigsaw(RotationPrediction):

    def __init__(self,
                 input_shape,
                 num_pretext_classes,
                 classifier=None,
                 tune_decoder_only=False):

        super(Jigsaw, self).__init__(
            input_shape=input_shape,
            classifier=classifier,
            tune_decoder_only=tune_decoder_only
        )

        self.pretext_classifier = tf.keras.Sequential(
            [
                tfkl.Flatten(),
                tfkl.Dense(num_pretext_classes, activation='softmax')
            ]
        )

    def call(self, x, training=False):

        x1, x2, x3, x4, x5 = self.encoder(x, training=training)
        output = self.decoder(
            [x1, x2, x3, x4, x5], training=training)
        if self.classifier is not None:
            output = self.classifier(output, training=training)
        return output

    def compile(self, ft_optimizer=None, **kwargs):
        super(Jigsaw, self).compile(ft_optimizer, **kwargs)

    def shared_step(self, data, training):

        x, y, x_shuffle, label = data

        features = self.encoder(x_shuffle, training=training)
        y_pred = self.pretext_classifier(features[-1])

        loss = self.loss_fn(label, y_pred)

        return features, loss

    def finetune_step(self, data):
        x, y, _, _ = data
        with tf.GradientTape() as tape:
            features = self.encoder(x, training=True)
            y_pred = self.classifier(
                self.decoder(features, training=True), training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        if self.tune_decoder_only:
            trainable_variables = self.decoder.trainable_variables + \
                self.classifier.trainable_variables
        else:
            trainable_variables = self.encoder.trainable_variables + \
                self.decoder.trainable_variables + \
                self.classifier.trainable_variables

        grads = tape.gradient(loss, trainable_variables)
        self.ft_optimizer.apply_gradients(zip(grads, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

    def test_step(self, data):

        _, loss = self.shared_step(data, training=False)

        if self.classifier is not None:
            x, y, _, _ = data
            features = self.encoder(x, training=False)
            y_pred = self.classifier(
                self.decoder(features, training=False))
            _ = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y, y_pred)
            metric_results = {m.name: m.result() for m in self.metrics}
            return {'pretext_loss': loss, **metric_results}
        else:
            return {'pretext_loss': loss}


def main(argv):

    del argv

    strategy = setup_accelerator(FLAGS.use_gpu,
                                 FLAGS.num_cores,
                                 FLAGS.tpu)

    sample_perms = sample_permutations(9)

    train_ds, val_ds = load_dataset(
            batch_size=FLAGS.batch_size,
            dataset_dir='gs://oai-challenge-dataset/tfrecords',
            training_mode='finetune',
            fraction_data=FLAGS.fraction_data,
            multi_class=FLAGS.multi_class,
            add_background=FLAGS.add_background,
            normalize=FLAGS.normalize,
            permutations=sample_perms,
            buffer_size=int(19200 * FLAGS.fraction_data),
            parse_fn=parse_fn_restore)

    steps_per_epoch = int(19200 * FLAGS.fraction_data) // FLAGS.batch_size
    validation_steps = 4480 // FLAGS.batch_size
    ds_shape = (288, 288, 1)

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

    with strategy.scope():

        if FLAGS.online_ft:
            classifier = tfkl.Conv2D(
                num_classes, (1, 1),
                activation=activation, padding='same')

        model = Jigsaw(
            input_shape=ds_shape,
            num_pretext_classes=1000,
            classifier=classifier,
            tune_decoder_only=FLAGS.finetune_decoder_only)

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

        if FLAGS.custom_schedule:
            lr_rate = LearningRateSchedule(steps_per_epoch,
                                           FLAGS.learning_rate,
                                           1e-09,
                                           0.8,
                                           list(range(1, FLAGS.train_epochs)),
                                           0)
        else:
            lr_rate = FLAGS.learning_rate

        optimizer = Adam(lr_rate)

        model.compile(ft_optimizer=Adam(FLAGS.ft_learning_rate),
                      optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = None

    if FLAGS.save_history:
        histdir = os.path.join(FLAGS.histdir, time)
        os.mkdir(histdir)

        # Create a callback for saving the training results into a csv file
        histfile = 'jigsaw_results.csv'
        csv_logger = CSVLogger(os.path.join(histdir, histfile))
        cb = [csv_logger]

        # Save flag params in a flag file in the same subdirectory
        flagfile = os.path.join(histdir, 'train_flags.cfg')
        FLAGS.append_flags_into_file(flagfile)

    model.fit(train_ds,
              steps_per_epoch=steps_per_epoch,
              epochs=FLAGS.train_epochs,
              validation_data=val_ds,
              validation_steps=validation_steps,
              verbose=1,
              callbacks=cb)


if __name__ == '__main__':
    app.run(main)
