import tensorflow as tf
import glob
import h5py
import os
import numpy as np
from functools import partial

from self_supervised.TF2.aug.oai_transform import get_preprocess_fn
AUTOTUNE = tf.data.AUTOTUNE


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an Eager Tensor
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float /p double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_OAI_challenge_dataset(data_folder,
                                 tfrecord_directory,
                                 get_train=True,
                                 use_2d=True):

    if not os.path.exists(tfrecord_directory):
        os.mkdir(tfrecord_directory)

    # train_val = 'train' if get_train else 'valid'
    files = glob(os.path.join(data_folder, f'*.im'))

    for idx, f in enumerate(files):
        f_name = f.split("/")[-1]
        f_name = f_name.split(".")[0]

        fname_img = f'{f_name}.im'
        fname_seg = f'{f_name}.seg'

        img_filepath = os.path.join(data_folder, fname_img)
        seg_filepath = os.path.join(data_folder, fname_seg)

        assert os.path.exists(seg_filepath), f"Seg file does not exist: {seg_filepath}"

        with h5py.File(img_filepath, 'r') as hf:
            img = np.array(hf['data'])
        with h5py.File(seg_filepath, 'r') as hf:
            seg = np.array(hf['data'])

        img = np.rollaxis(img, 2, 0)
        img = np.expand_dims(img, axis=-1)
        seg = np.rollaxis(seg, 2, 0)

        assert img.shape[-1] == 1
        assert seg.shape[-1] == 6

        shard_dir = f'{idx:03d}-of-{len(files) - 1:03d}.tfrecords'
        tfrecord_filename = os.path.join(tfrecord_directory, shard_dir)

        target_shape, label_shape = None, None
        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            if use_2d:
                for k in range(len(img)):
                    img_slice = img[k, :, :, :]
                    seg_slice = seg[k, :, :, :]

                    img_raw = img_slice.tostring()
                    seg_raw = seg_slice.tostring()

                    height = img_slice.shape[0]
                    width = img_slice.shape[1]
                    num_channels = seg_slice.shape[-1]

                    target_shape = img_slice.shape
                    label_shape = seg.shape

                    feature = {
                        'height': _int64_feature(height),
                        'width': _int64_feature(width),
                        'num_channels': _int64_feature(num_channels),
                        'image_raw': _bytes_feature(img_raw),
                        'label_raw': _bytes_feature(seg_raw)
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            else:
                height = img.shape[0]
                width = img.shape[1]
                depth = img.shape[2]
                num_channels = seg.shape[-1]

                target_shape = img.shape
                label_shape = seg.shape

                img_raw = img.tostring()
                seg_raw = seg.tostring()

                feature = {
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'depth': _int64_feature(depth),
                    'num_channels': _int64_feature(num_channels),
                    'image_raw': _bytes_feature(img_raw),
                    'label_raw': _bytes_feature(seg_raw)
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print(f'{idx+1} out of {len(files)} datasets have been processed. Target: {target_shape}, Label: {label_shape}')


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


if __name__ == '__main__':

    train_ds, val_ds = load_dataset(
        batch_size=256,
        dataset_dir='gs://oai-challenge-dataset/tfrecords/',
        training_mode='pretrain')

    for image, label in train_ds:
        print(image.shape)
        print(label.shape)
