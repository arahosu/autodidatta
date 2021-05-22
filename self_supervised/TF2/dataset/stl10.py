import tensorflow as tf
import tensorflow_datasets as tfds

from self_supervised.TF2.models.simclr.simclr_transforms import get_preprocess_fn

def load_input_fn(split,
                  batch_size,
                  training_mode,
                  drop_remainder=True):

    """STL10 dataset loader for training or testing.
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

    name = 'stl10'
    ds_info = tfds.builder(name).info
    
    if (split == 'train') or split == 'unlabelled':
        is_training = True
    else:
        is_training = False

    preprocess_fn_pretrain = get_preprocess_fn(is_training=is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training=is_training, is_pretrain=False)

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        label = tf.cast(label, tf.float32)

        if training_mode == 'pretrain':
            xs = []
            for _ in range(2):
                xs.append(preprocess_fn_pretrain(image))
            image = tf.concat(xs, -1)
        else:
            image = preprocess_fn_finetune(image)
        return image, label

    dataset = tfds.load(name, split=split, as_supervised=True)

    """
    if (split == 'train') or (split == 'unlabelled'):
        dataset = dataset.shuffle(buffer_size=1000).repeat()
    else:
        dataset = dataset.repeat()
    """

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset    

    