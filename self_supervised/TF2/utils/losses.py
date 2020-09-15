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

def nt_xent_loss_v2(out_i, out_j, temperature):
    
    # positive similarity 

    
    loss = 0
    negatives = tf.concat([out_i, out_j], axis=0)
    n_samples = negatives.shape[0]

    for positives in [out_i, out_j]:
        l_neg = tf.tensordot(tf.expand_dims(positives, 1), tf.expand_dims(tf.transpose(negatives), 1), axes=2)
        labels = tf.zeros(n_samples / 2, dtype=tf.int32)

