import tensorflow as tf
import tensorflow_datasets as tfds

from self_supervised.TF2.models.simclr.simclr_transforms import get_preprocess_fn

def load_input_fn(split,
                  batch_size,
                  name,
                  training_mode,
                  use_cloud=True,
                  normalize=False,
                  drop_remainder=True,
                  proportion=1.0):

    """Prototype CIFAR dataset loader for training or testing.
    https://github.com/google/uncertainty-baselines/blob/master/baselines/cifar/utils.py
    Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    normalize: Whether to apply mean-std normalization on features.
    drop_remainder: bool.
    proportion: float, the proportion of dataset to be used.
    Returns:
    Input function which returns a locally-sharded dataset batch.
    """
    ds_info = tfds.builder(name).info
    image_shape = ds_info.features['image'].shape
    dataset_size = ds_info.splits['train'].num_examples

    if split == tfds.Split.TRAIN:
        is_training = True
    else:
        is_training = False

    preprocess_fn_pretrain = get_preprocess_fn(is_training=is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training=is_training, is_pretrain=False)

    def preprocess(image, label):
        """Image preprocessing function. Augmentations should be written
        as a separate function"""

        image = tf.image.convert_image_dtype(image, tf.float32)  # THIS STEP IS CRITICAL. DO NOT USE tf.cast
        label = tf.cast(label, tf.float32)

        if normalize:
            mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
            std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
            image = (image - mean) / std
        if training_mode == 'pretrain':
            xs = []
            for _ in range(2):
                xs.append(preprocess_fn_pretrain(image))
            image = tf.concat(xs, -1)
        else:
            image = preprocess_fn_finetune(image)

        return image, label

    if proportion == 1.0:
        if use_cloud:
            dataset = tfds.load(name, split=split, data_dir='gs://cifar10_baseline/', as_supervised=True)
        else:
            dataset = tfds.load(name, split=split, data_dir='C:/Users/Joonsu/tensorflow_datasets/', as_supervised=True)
    else:
        new_name = '{}:3.*.*'.format(name)
        if split == tfds.Split.TRAIN:
            new_split = 'train[:{}%]'.format(int(100 * proportion))
        else:
            new_split = 'test[:{}%]'.format(int(100 * proportion))
        if use_cloud:
            dataset = tfds.load(new_name, split=new_split, data_dir='gs://cifar10_baseline/', as_supervised=True)
        else:
            dataset = tfds.load(new_name, split=new_split, data_dir='C:/Users/Joonsu/tensorflow_datasets/', as_supervised=True)

    if split == tfds.Split.TRAIN:
        dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
