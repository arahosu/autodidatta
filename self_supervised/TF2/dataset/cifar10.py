import tensorflow as tf
import tensorflow_datasets as tfds

from self_supervised.TF2.models.simclr.simclr_transforms import get_preprocess_fn

def load_input_fn(split,
                  batch_size,
                  training_mode,
                  normalize=False,
                  drop_remainder=True,
                  proportion=1.0):

    """CIFAR10 dataset loader for training or testing.
    https://github.com/google/uncertainty-baselines/blob/master/baselines/cifar/utils.py
    Args:
    split (str): 'train' for training split, 'test' for test split
    batch_size (int): The global batch size to use.
    training_mode (str): 'pretrain' for training, 'finetune' for fine-tuning
    normalize (bool): Whether to apply mean-std normalization on features.
    drop_remainder (bool): Whether to drop the last batch if it has fewer than batch_size elements
    proportion (float): The proportion of dataset to be used.
    Returns:
    Input function which returns a cifar10 dataset.
    """

    name = 'cifar10'
    ds_info = tfds.builder(name).info
    dataset_size = ds_info.splits['train'].num_examples

    if split == 'train':
        is_training = True
    else:
        is_training = False

    preprocess_fn_pretrain = get_preprocess_fn(is_training=is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training=is_training, is_pretrain=False)

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
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

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if split == 'train':
        num_examples = int(len(x_train) * proportion)
        (x_train_split, y_train_split) = (x_train[:num_examples,...], y_train[:num_examples,:])
        dataset = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split))
    else:
        num_examples = int(len(x_test) * proportion)
        (x_test_split, y_test_split) = (x_test[:num_examples,...], y_test[:num_examples,:])
        dataset = tf.data.Dataset.from_tensor_slices((x_test_split, y_test_split))

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
