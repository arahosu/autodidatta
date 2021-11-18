import autodidatta.augment as A
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets.cifar10 import load_data

AUTOTUNE = tf.data.AUTOTUNE


def load_input_fn(is_training,
                  batch_size,
                  image_size,
                  aug_fn,
                  aug_fn_2=None,
                  pre_train=True,
                  drop_remainder=True,
                  proportion=1.0,
                  use_bfloat16=True):

    """CIFAR10 dataset loader for training or testing.
    Args:
    is_training (bool): True for training split, False for validation split
    batch_size (int): The global batch size to use.
    image_size (int): The image size to use
    aug_fn (A.Augment, tf.keras.Sequential): Augmentation function
    aug_fn_2 (A.Augment, tf.keras.Sequential): Optional 2nd Augmentation function 
    pre_train (bool): True for pre-training, False for finetuning
    drop_remainder (bool): Whether to drop the last batch if it has fewer than
    batch_size elements
    proportion (float): The proportion of training images to be used.
    use_bfloat16 (bool): True if set dtype to bfloat16, False to set dtype to float32
    Returns:
    Input function which returns a cifar10 dataset.
    """

    name = 'cifar10'
    ds_info = tfds.builder(name).info
    dataset_size = ds_info.splits['train'].num_examples

    def preprocess(image, label):
        dtype = tf.bfloat16 if use_bfloat16 else tf.float32
        label = tf.cast(label, dtype)

        if pre_train:
            xs = []
            for i in range(2):
                augmentation_fn = aug_fn
                if aug_fn_2 is not None:
                    augmentation_fn = aug_fn_2 if i == 1 else aug_fn
                aug_img = augmentation_fn(image, training=is_training)
                # aug_img = tf.clip_by_value(aug_img, 0., 1.)
                aug_img = tf.reshape(aug_img, [image_size, image_size, 3])
                xs.append(aug_img)
            image = tf.concat(xs, -1)
        else:
            if aug_fn is not None:
                image = aug_fn(image, training=is_training)
            # image = tf.clip_by_value(image, 0., 1.)
            image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.cast(image, dtype)
        return image, label

    (x_train, y_train), (x_test, y_test) = load_data()
    if is_training:
        num_examples = int(len(x_train) * proportion)
        (x_train_split, y_train_split) = (x_train[:num_examples, ...],
                                          y_train[:num_examples, :])
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train_split, y_train_split))
    else:
        num_examples = int(len(x_test) * proportion)
        (x_test_split, y_test_split) = (x_test[:num_examples, ...],
                                        y_test[:num_examples, :])
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_test_split, y_test_split))

    if is_training:
        dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
