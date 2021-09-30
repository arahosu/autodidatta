import tensorflow as tf
import tensorflow_io as tfio

import time
import datetime

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

            if tf.rank(image) == 4:
                counter += 1

            if idx % 100 == 0:
                print('{} out of {} images processed'.format(idx+1, num_files))

        return counter
    else:
        return num_files


if __name__ == '__main__':

    convert_dicom_to_tfrecords("dicom_files.txt", "00m", "01-of-09.tfrecords")
