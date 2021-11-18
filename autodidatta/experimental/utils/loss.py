import tensorflow as tf
from autodidatta.utils.loss import tpu_cross_replica_concat

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


def invariance_variance_loss(hidden1, hidden2, mu, nu, strategy=None):

    if strategy is not None:
        hidden1 = tpu_cross_replica_concat(hidden1, strategy)
        hidden2 = tpu_cross_replica_concat(hidden2, strategy)

    # invariance loss
    MSE_loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    sim_loss = MSE_loss(hidden1, hidden2)

    # variance loss
    std_hidden1 = tf.math.sqrt(tf.math.reduce_variance(hidden1, axis=0) + 1e-04)
    std_hidden2 = tf.math.sqrt(tf.math.reduce_variance(hidden2, axis=0) + 1e-04)
    hidden_std = (std_hidden1 + std_hidden2) / 2

    std_loss = tf.math.reduce_mean(tf.nn.relu(1 - std_hidden1)) + \
        tf.math.reduce_mean(tf.nn.relu(1 - std_hidden2))

    # total loss
    loss = mu * sim_loss + nu * std_loss
    loss /= strategy.num_replicas_in_sync

    return loss

    
        