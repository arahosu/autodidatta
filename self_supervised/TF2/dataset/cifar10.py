import tensorflow as tf
import tensorflow_datasets as tfds

def load_input_fn(split,
                  batch_size,
                  name,
                  use_bfloat16,
                  normalize=True,
                  drop_remainder=True,
                  proportion=1.0):
    """Prototype CIFAR dataset loader for training or testing.
    https://github.com/google/uncertainty-baselines/blob/master/baselines/cifar/utils.py
    Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    use_bfloat16: data type, bfloat16 precision or float32.
    normalize: Whether to apply mean-std normalization on features.
    drop_remainder: bool.
    proportion: float, the proportion of dataset to be used.
    Returns:
    Input function which returns a locally-sharded dataset batch.
    """
    if use_bfloat16:
        dtype = tf.bfloat16
    else:
        dtype = tf.float32
    ds_info = tfds.builder(name).info
    image_shape = ds_info.features['image'].shape
    dataset_size = ds_info.splits['train'].num_examples

    def preprocess(image, label):
        """Image preprocessing function. Augmentations should be written
        as a separate function"""

        # if split == tfds.Split.TRAIN:
        #   image = tf.image.resize_with_crop_or_pad(image, image_shape[0] + 4, image_shape[1] + 4)
        #   image = tf.image.random_crop(image, image_shape)
        #   image = tf.image.random_flip_left_right(image)

        image = tf.image.convert_image_dtype(image, dtype)
        if normalize:
            mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
            std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
            image = (image - mean) / std
        label = tf.cast(label, dtype)
        return image, label

    def input_fn(ctx=None):
        """Returns a locally sharded (i.e., per-core) dataset batch."""
        if proportion == 1.0:
            dataset = tfds.load(name, split=split, as_supervised=True)
        else:
            new_name = '{}:3.*.*'.format(name)
            if split == tfds.Split.TRAIN:
                new_split = 'train[:{}%]'.format(int(100 * proportion))
            else:
                new_split = 'test[:{}%]'.format(int(100 * proportion))
        dataset = tfds.load(new_name, split=new_split, as_supervised=True)
        if split == tfds.Split.TRAIN:
            dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if ctx and ctx.num_input_pipelines > 1:
            dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        return dataset
    return input_fn
