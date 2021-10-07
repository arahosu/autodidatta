import tensorflow as tf


def tversky_loss(y_true,
                 y_pred,
                 alpha=0.5,
                 beta=0.5,
                 smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : tensor containing target mask.
    y_pred : tensor containing predicted mask.
    alpha : real value, weight of '0' class.
    beta : real value, weight of '1' class.
    smooth : small real value used for avoiding division by zero error.
    Returns
    -------
    tensor
        tensor containing tversky loss.
    """

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    truepos = tf.math.reduce_sum(y_true * y_pred)
    fp_and_fn = alpha * tf.math.reduce_sum(
        y_pred * (1 - y_true)) + beta * tf.math.reduce_sum(
        (1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)

    return tf.cast(1 - answer, tf.float32)
