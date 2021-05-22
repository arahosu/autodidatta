import tensorflow as tf
import os
from functools import partial

from sss.augmentation.dual_transform import get_preprocess_fn
AUTOTUNE = tf.data.AUTOTUNE


def add_background(image):

    image_not_background = tf.clip_by_value(
        tf.math.reduce_sum(image, axis=-1), 0, 1)
    image_background = tf.expand_dims(
        tf.math.logical_not(
            tf.cast(image_not_background, dtype=tf.bool)), axis=-1)
    image_background = tf.cast(image_background, dtype=tf.float32)
    return tf.concat([image_background, image], axis=-1)


def parse_fn_2d(example_proto,
                training_mode,
                is_training):

    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.float32)
    image = tf.cast(tf.reshape(image_raw, [384, 384, 1]), tf.float32)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int8)
    seg = tf.reshape(seg_raw, [384, 384, 6])
    seg = tf.cast(seg, tf.float32)

    tf.debugging.check_numerics(image, "Invalid value in your input!")
    tf.debugging.check_numerics(seg, "Invalid value in your label!")

    preprocess_fn_pretrain = get_preprocess_fn(
        is_training=is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(
        is_training=is_training, is_pretrain=False)

    if training_mode == 'pretrain':
        image1, seg1 = preprocess_fn_pretrain(
            image=image, mask=seg)
        image2, seg2 = preprocess_fn_pretrain(
            image=image, mask=seg)

        image = tf.concat([image1, image2], -1)
        # seg = tf.concat([add_background(seg1), add_background(seg2)], -1)
        seg = tf.concat([seg1, seg2], -1)
        return (image, seg)
    else:
        image, seg = preprocess_fn_finetune(
            image=image, mask=seg)
        # seg = add_background(seg)
        return (image, seg)


def read_tfrecord(tfrecords_dir,
                  batch_size,
                  buffer_size,
                  is_training,
                  training_mode,
                  parse_fn=parse_fn_2d):

    """This function reads and returns TFRecords dataset in tf.data.Dataset format
    """

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)
    if is_training:
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
        cycle_length = 8
    else:
        cycle_length = 1

    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=cycle_length,
                                num_parallel_calls=AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(
        partial(
            parse_fn, training_mode=training_mode, is_training=is_training),
        num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    # optimise dataset performance
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_dataset(batch_size,
                 dataset_dir,
                 training_mode,
                 buffer_size=1000):

    """Function for loading and parsing dataset
    """

    # Define the directories of the training and validation files
    train_dir, valid_dir = 'train/', 'valid/'

    # Define the datasets as tf.data.Datasets using read_tfrecord function
    train_ds = read_tfrecord(
        tfrecords_dir=os.path.join(dataset_dir, train_dir),
        batch_size=batch_size,
        buffer_size=buffer_size,
        is_training=True,
        training_mode=training_mode
        )

    valid_ds = read_tfrecord(
        tfrecords_dir=os.path.join(dataset_dir, valid_dir),
        batch_size=batch_size,
        buffer_size=buffer_size,
        is_training=False,
        training_mode=training_mode
        )

    return train_ds, valid_ds
