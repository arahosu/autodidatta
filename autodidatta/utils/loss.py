import tensorflow as tf
LARGE_NUM = 1e9


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


def tpu_cross_replica_concat(tensor, strategy=None):
    """Reduce a concatenation of the `tensor` across TPU cores.
    Args:
    tensor: tensor to concatenate.
    strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.
    Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
    """
    if strategy is None or strategy.num_replicas_in_sync <= 1:
        return tensor

    num_replicas = strategy.num_replicas_in_sync

    replica_context = tf.distribute.get_replica_context()
    with tf.name_scope('tpu_cross_replica_concat'):
        # This creates a tensor that is like the input tensor but has an added
        # replica dimension as the outermost dimension. On each replica it will
        # contain the local values and zeros for all other values that need to
        # be fetched from other replicas.
        ext_tensor = tf.scatter_nd(
            indices=[[replica_context.replica_id_in_sync_group]],
            updates=[tensor],
            shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

        # As every value is only present on one replica and 0 in all others,
        # adding them all together will result in the full tensor on all
        # replicas.
        ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                                ext_tensor)

        # Flatten the replica dimension.
        # The first dimension size will be: tensor.shape[0] * num_replicas
        # Using [-1] trick to support also scalar input.
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])


def nt_xent_loss_v2(hidden1,
                    hidden2,
                    temperature=0.5,
                    strategy=None):

    """Compute loss for model.
    Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    temperature: a `floating` number for temperature scaling.
    strategy: context information for tpu.
    Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    batch_size = tf.shape(hidden1)[0]

    # Gather hidden1/hidden2 across replicas and create local labels.
    if strategy is not None:
        hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
        hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
        enlarged_batch_size = tf.shape(hidden1_large)[0]
        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(
            tf.cast(
                replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
        masks = tf.one_hot(labels_idx, enlarged_batch_size)
    else:
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(
        hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(
        hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(
        hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(
        hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)

    loss /= strategy.num_replicas_in_sync

    return loss


def cosine_similarity_v2(hidden1,
                         hidden2,
                         temperature=1.0,
                         stop_gradient=True,
                         strategy=None):

    if strategy is not None:
        hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
        hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
    else:
        hidden1_large = hidden1
        hidden2_large = hidden2

    if stop_gradient:
        return temperature * tf.keras.losses.cosine_similarity(
            hidden1_large, tf.stop_gradient(hidden2_large))
    else:
        return temperature * tf.keras.losses.cosine_similarity(
            hidden1_large, hidden2_large)


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