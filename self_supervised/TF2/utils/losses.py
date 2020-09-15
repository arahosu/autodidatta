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
    neg = tf.math.reduce_sum(tf.reshape(tf.boolean_mask(sim, mask), [n_samples, -1]), axis=-1)

    # Positive Similiarity
    pos = tf.math.exp(tf.math.reduce_sum(out_i * out_j, axis=-1) / temperature)
    pos = tf.concat([pos, pos], axis=0)
    loss = tf.math.reduce_mean(-tf.math.log(pos / neg))

    return loss


def mse_loss(online_network_out_1, online_network_out_2, target_network_out_1, target_network_out_2):
    """ Compute BYOLs loss function. Mean square error between
    the normalized predictions and target projections.
    Args:
        online_network_out_1: prediction head output of online network on sample 1
        online_network_out_2: prediction head output of online network on sample 2
        online_network_out_1: projection head output of target network on sample 1
        online_network_out_1: projection head output of target network on sample 2
    """

    def regression_loss(x, y):
        norm_x, norm_y = tf.norm(x), tf.norm(y)
        return (-2. * tf.keras.backend.sum(x * y, axis=-1) / (norm_x * norm_y))

    # TODO: Add stop gradient to target networks?
    loss = regression_loss(online_network_out_1, target_network_out_2)
    loss += regression_loss(online_network_out_2, target_network_out_1)

    return loss
