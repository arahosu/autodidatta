import tensorflow as tf
import tensorflow_io as tfio

import time
import datetime
import os
import numpy as np
from functools import partial

from sss.augmentation.dual_transform import get_preprocess_fn
from sss.utils import min_max_standardize
AUTOTUNE = tf.data.AUTOTUNE


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
        # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def parse_fn(example_proto,
             is_training,
             normalize):

    features = {
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'patient_id': tf.io.FixedLenFeature([], tf.int64)
                }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.int16)
    image = tf.cast(tf.reshape(image_raw, [384, 384, 1]), tf.float32)
    patient_id = image_features['patient_id']

    if normalize:
        image = min_max_standardize(image)

    preprocess_fn = get_preprocess_fn(is_training, True, 288)

    image1, _ = preprocess_fn(image, mask=None)
    image2, _ = preprocess_fn(image, mask=None)

    image = tf.concat([image1, image2], -1)
    return image


def get_num_examples(tfrecords_dir):

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))

    for file in file_list:

        print(file)
        dataset = tf.data.TFRecordDataset(file)
        dataset = dataset.map(
            partial(
                parse_fn,
                is_training=True,
                normalize=True),
            num_parallel_calls=AUTOTUNE)

        dataset = dataset.batch(1, drop_remainder=True).prefetch(AUTOTUNE)

        cnt = dataset.reduce(np.int64(0), lambda x, _: x + 1)
        print(cnt)


def read_tfrecord(tfrecords_dir,
                  is_training,
                  batch_size,
                  normalize,
                  buffer_size,
                  drop_remainder=True):

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)

    get_num_examples(file_list)

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
                parse_fn,
                is_training=is_training,
                normalize=normalize),
            num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    return dataset


def load_oai_full_dataset(tfrecords_dir,
                          batch_size,
                          normalize,
                          buffer_size):

    train_ds = read_tfrecord(tfrecords_dir,
                             True,
                             batch_size,
                             normalize,
                             buffer_size)

    return train_ds


def count_dicom_files(text_file, keyword, count_valid_examples=False):

    # filter the text file according to keyword
    text_file = open(text_file, "r")
    lines = text_file.readlines()
    filelist = []
    for filename in lines:
        line = filename[:-1]
        if keyword in line:
            filelist.append(line)
        elif keyword is None:
            filelist.append(line)

    num_files = len(filelist)
    print('Number of DICOM files detected: {}'.format(num_files))

    if count_valid_examples:
        counter = 0

        for idx, f in enumerate(filelist):
            image_bytes = tf.io.read_file(f)
            image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
            image = image.numpy()
            if image.ndim == 4:
                image = image = image[0, ...]

            if image.ndim == 3 and image.shape == (384, 384, 1):
                counter += 1

            if idx % 1000 == 0:
                print('{} out of {} images processed'.format(idx+1, num_files))

        return counter
    else:
        return num_files


def convert_dicom_to_tfrecords(text_file, keyword, dest_file):
    # filter the text file according to keyword
    text_file = open(text_file, "r")
    lines = text_file.readlines()
    filelist = []
    for filename in lines:
        line = filename[:-1]
        if keyword in line:
            filelist.append(line)
        elif keyword is None:
            filelist.append(line)

    num_files = len(filelist)
    start_time = time.time()

    with tf.io.TFRecordWriter(dest_file) as writer:
        for idx, f in enumerate(filelist):
            image_bytes = tf.io.read_file(f)
            image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
            image = image.numpy()
            image = image[0, ...]
            print(image.shape)

            if image.ndim == 4:

                image_raw = image.tobytes()
                patientid = tfio.image.decode_dicom_data(
                    image_bytes, tags=tfio.image.dicom_tags.PatientID)
                patientid = tf.strings.to_number(patientid, tf.int64)

                feature = {
                    'image_raw': _bytes_feature(image_raw),
                    'patient_id': _int64_feature(patientid)

                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

                current_time = time.time()
                elapsed_time = current_time - start_time
                approx_total_time = elapsed_time * (num_files / (idx + 1))
                remaining_time = int(approx_total_time - elapsed_time)

                if idx % 100 == 0:
                    print(f'{idx+1} out of {num_files} images processed.')
                    time_remaining = str(
                        datetime.timedelta(seconds=remaining_time))
                    print('remaining time: {}'.format(time_remaining))
            else:
                print('Invalid DICOM file found at idx: {}'.format(idx))


if __name__ == '__main__':
    # convert_dicom_to_tfrecords("dicom_files.txt", "00m", "01-of-09.tfrecords")
    count_dicom_files("dicom_files.txt", "00m", count_valid_examples=True)

