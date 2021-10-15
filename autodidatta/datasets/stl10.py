import autodidatta.augment as A
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


# TODO(arahosu): allow users to control the proportion of images to be used 
def load_input_fn(is_training,
                  batch_size,
                  image_size,
                  pre_train,
                  aug_fn,
                  aug_fn_2=None,
                  drop_remainder=True,
                  proportion=1.0):

    """STL10 dataset loader for training or testing.
    Args:
    is_training (bool): True for training split, False for validation split
    batch_size (int): The global batch size to use.
    image_size (int): The image size to use
    pre_train (bool): True for pre-training, False for finetuning
    aug_fn (A.Augment, tf.keras.Sequential): Augmentation function
    aug_fn_2 (A.Augment, tf.keras.Sequential): Optional 2nd Augmentation function 
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

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        if not pre_train:
            label = tf.cast(label, tf.float32)

        if pre_train:
            xs = []
            for i in range(2):
                augmentation_fn = aug_fn
                if aug_fn_2 is not None:
                    augmentation_fn = aug_fn_2 if i == 1 else aug_fn
                aug_img = augmentation_fn(image, training=is_training)
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