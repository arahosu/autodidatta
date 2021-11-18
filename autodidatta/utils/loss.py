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
    """
    Taken from Google Research's SimCLR repository:
    https://github.com/google-research/simclr/blob/master/tf2/objective.py

    Reduce a concatenation of the `tensor` across TPU cores.
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

    """
    Taken from Google Research's SimCLR repository:
    https://github.com/google-research/simclr/blob/master/tf2/objective.py

    Compute loss for model.
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

    labels = tf.cast(labels, hidden1.dtype)
    masks = tf.cast(masks, hidden1.dtype)

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


def byol_loss(hidden1,
              hidden2,
              strategy=None):

    hidden1 = tf.math.l2_normalize(hidden1, axis=-1)
    hidden2 = tf.math.l2_normalize(hidden2, axis=-1)

    loss = 2 - 2 * tf.math.reduce_mean(tf.math.reduce_sum(hidden1 * hidden2, axis=-1))

    loss /= strategy.num_replicas_in_sync

    return loss


def barlow_twins_loss(hidden1,
                      hidden2,
                      lambda_,
                      loss_temperature=0.025,
                      strategy=None):
    dtype = hidden1.dtype

    if strategy is not None:
        hidden1 = tpu_cross_replica_concat(hidden1, strategy=strategy)
        hidden2 = tpu_cross_replica_concat(hidden2, strategy=strategy)

    N, D = hidden1.shape[0], hidden2.shape[1]
    N = tf.cast(N, dtype)

    # normalize repr. along the batch dimension
    zi_norm = (hidden1 - tf.reduce_mean(hidden1, axis=0)) / \
        tf.math.reduce_std(hidden1, axis=0)  # (b, i)
    zj_norm = (hidden2 - tf.reduce_mean(hidden2, axis=0)) / \
        tf.math.reduce_std(hidden2, axis=0)  # (b, j)

    # cross-correlation matrix
    # c_ij = tf.einsum('bi,bj->ij',
    #                  tf.math.l2_normalize(zi_norm, axis=0),
    #                  tf.math.l2_normalize(zj_norm, axis=0)) / N  # (i, j)
    c_ij = tf.matmul(zi_norm, zj_norm, transpose_a=True) / N

    # for separating invariance and reduction
    loss_invariance = tf.reduce_sum(
        tf.square(1. - tf.boolean_mask(c_ij, tf.eye(D, dtype=tf.bool))))
    loss_reduction = tf.reduce_sum(
        tf.square(tf.boolean_mask(c_ij, ~tf.eye(D, dtype=tf.bool))))

    loss = loss_invariance + lambda_ * loss_reduction

    # on_diag = tf.linalg.diag_part(c_ij) + (-1)
    # on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    # off_diag = off_diagonal(c_ij)
    # off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    # loss = on_diag + (lambda_ * off_diag)

    loss *= loss_temperature
    loss /= strategy.num_replicas_in_sync

    return loss