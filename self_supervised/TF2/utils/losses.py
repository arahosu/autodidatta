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

def mse_loss(prediction_q, target_proj_z):
    """ Mean square error between the normalized predictions
    and target projections """

    expected_value = 1
    normalized_q = 1
    normalized_z = 1
    loss = 2 - 2 * expected_value / normalized_q * normalized_z

    return loss