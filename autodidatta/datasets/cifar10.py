import autodidatta.augment as A
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets.cifar10 import load_data

AUTOTUNE = tf.data.AUTOTUNE


def aug_fn(image, image_size, is_training, pre_train):

    color_jitter_prob = 0.8 if pre_train else 0.
    grayscale_prob = 0.2 if pre_train else 0.

    transforms = A.Augment([
            A.layers.RandomResizedCrop(image_size, image_size),
            A.layers.ColorJitter(0.4, 0.4, 0.4, 0.1, p=color_jitter_prob),
            A.layers.ToGray(p=grayscale_prob)
            ])

    # apply augmentation
    aug_img = transforms(image, training=is_training)
    return aug_img


def load_input_fn(is_training,
                  batch_size,
                  image_size,
                  pre_train,
                  drop_remainder=True,
                  proportion=1.0):

    """CIFAR10 dataset loader for training or testing.
    Args:
    is_training (bool): True for training split, False for validation split
    batch_size (int): The global batch size to use.
    image_size (int): The image size to use
    pre_train (bool): True for pre-training, False for finetuning
    drop_remainder (bool): Whether to drop the last batch if it has fewer than
    batch_size elements
    proportion (float): The proportion of training images to be used.
    Returns:
    Input function which returns a cifar10 dataset.
    """

    name = 'cifar10'
    ds_info = tfds.builder(name).info
    dataset_size = ds_info.splits['train'].num_examples

    preprocess_fn = partial(
        aug_fn,
        image_size=image_size, is_training=is_training, pre_train=pre_train)

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.cast(label, tf.float32)

        if pre_train:
            xs = []
            for _ in range(2):
                aug_img = preprocess_fn(image)
                aug_img = tf.clip_by_value(aug_img, 0., 1.)
                aug_img = tf.reshape(aug_img, [image_size, image_size, 3])
                xs.append(aug_img)
            image = tf.concat(xs, -1)
        else:
            image = preprocess_fn(image)
            image = tf.clip_by_value(image, 0., 1.)
            image = tf.reshape(image, [image_size, image_size, 3])
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
