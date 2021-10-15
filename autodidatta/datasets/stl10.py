import autodidatta.augment as A
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds

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

# TODO(arahosu): allow users to control the proportion of images to be used 
def load_input_fn(is_training,
                  batch_size,
                  image_size,
                  pre_train,
                  drop_remainder=True,
                  proportion=1.0):

    """STL10 dataset loader for training or testing.
    Args:
    is_training (bool): True for training split, False for validation split
    batch_size (int): The global batch size to use.
    image_size (int): The image size to use
    pre_train (bool): True for pre-training, False for finetuning
    drop_remainder (bool): Whether to drop the last batch if it has fewer than
    batch_size elements
    proportion (float): The proportion of training images to be used.
    Returns:
    Input function which returns a stl10 dataset.
    """

    name = 'stl10'
    ds_info = tfds.builder(name).info
    if is_training:
        if pre_train:
            split = 'unlabelled'
        else:
            split = 'train'
    else:
        split = 'test'
    dataset_size = ds_info.splits[split].num_examples

    preprocess_fn = partial(
        aug_fn,
        image_size=image_size, is_training=is_training, pre_train=pre_train)

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        if not pre_train:
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
        
        if pre_train:
            return image
        else:
            return image, label

    dataset = tfds.load(name,
                        split=split,
                        shuffle_files=True if is_training else False,
                        as_supervised=True if pre_train else True)

    if is_training:
        dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


if __name__ == '__main__':
    train_ds = load_input_fn(is_training=True,
                             batch_size=512,
                             image_size=96,
                             pre_train=True)

    for data in train_ds:
        if isinstance(data, tuple):
            image, label = data
        else:
            image = data
        print(image.shape)
        break