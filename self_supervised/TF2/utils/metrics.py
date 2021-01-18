import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f * y_true_f) + tf.math.reduce_sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_single(y_true, y_pred, idx):

    y_true = y_true[..., idx]
    y_pred = y_pred[..., idx]

    return dice_coef(y_true, y_pred)


class DiceMetrics(MeanMetricWrapper):
    def __init__(self, idx, dtype=None):
        name = 'dice_coef_{}'.format(idx)
        super(DiceMetrics, self).__init__(
            dice_coef_single, name, dtype=dtype, idx=idx)


def dice_coef_eval(y_true, y_pred):

    # remove background_classes
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]

    return dice_coef(y_true, y_pred)