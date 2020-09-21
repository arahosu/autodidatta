import tensorflow as tf
import numpy as np

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

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

def nt_xent_loss_v2(out_i, out_j, temperature):

    loss = 0
    negatives = tf.concat([out_i, out_j], axis=0)
    n_samples = negatives.shape[0]

    # positive similarity
    l_pos = tf.matmul(tf.expand_dims(out_i, 1), tf.expand_dims(out_j, 2))
    l_pos = tf.reshape(l_pos, (n_samples // 2, 1))
    l_pos /= temperature

    for positives in [out_i, out_j]:
        l_neg = tf.tensordot(tf.expand_dims(positives, 1), tf.expand_dims(tf.transpose(negatives), 1), axes=2)
        labels = tf.zeros(n_samples // 2, dtype=tf.int32)

        l_neg = tf.boolean_mask(l_neg, get_negative_mask(n_samples // 2))
        l_neg = tf.reshape(l_neg, (n_samples // 2, -1))
        l_neg /= temperature

        logits = tf.concat([l_pos, l_neg], axis=1)
        loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    loss = loss / n_samples

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
