import tensorflow as tf


def nt_xent_loss(out_i, out_j, temperature):
    """Negative cross-entropy loss function for SimCLR
    """
    out = tf.concat([out_i, out_j], axis=0)
    n_samples = out.shape[0]

    cov = tf.linalg.matmul(out, tf.transpose(out))
    sim = tf.math.exp(cov / temperature)

    # Negative Similarity
    identity = tf.cast(tf.eye(n_samples), dtype=tf.bool)
    mask = tf.math.logical_not(identity)
    neg = tf.math.reduce_sum(
        tf.reshape(tf.boolean_mask(sim, mask), [n_samples, -1]), axis=-1)

    # Positive Similiarity
    pos = tf.math.exp(
        tf.math.reduce_sum(out_i * out_j, axis=-1) / temperature)
    pos = tf.concat([pos, pos], axis=0)

    # Loss
    loss = tf.math.reduce_mean(-tf.math.log(pos / neg))

    return loss


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

    # y_true = y_true[..., 1:]
    # y_pred = y_pred[..., 1:]

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    truepos = tf.math.reduce_sum(y_true * y_pred)
    fp_and_fn = alpha * tf.math.reduce_sum(
        y_pred * (1 - y_true)) + beta * tf.math.reduce_sum(
        (1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)

    return tf.cast(1 - answer, tf.float32)
