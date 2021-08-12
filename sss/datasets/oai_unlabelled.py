import tensorflow as tf
import tensorflow_io as tfio

import time
import datetime

from sss.augmentation.dual_transform import get_preprocess_fn
from sss.utils import min_max_standardize
AUTOTUNE = tf.data.AUTOTUNE


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  


def load_oai_full_dataset(text_file,
                          batch_size,
                          image_size,
                          train_split=0.640,
                          drop_remainder=True):

    # save the text file in a list
    text_file = open(text_file, "r")
    lines = text_file.readlines()
    filelist = []
    for filename in lines:
        line = filename[:-1]
        filelist.append(line)

    # split the filelist into training and validation filelists
    num_train = int(len(filelist)*train_split)
    train_list = filelist[-num_train:]
    val_list = filelist[:-num_train]

    def read_dicom_file(line, is_training):
        image_bytes = tf.io.read_file(line)
        image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
        image = tf.cast(image, tf.float32)
        image = min_max_standardize(image)
        image = tf.reshape(image, [384, 384, 1])
        preprocess_fn = get_preprocess_fn(is_training, True, 288)
        image1, _ = preprocess_fn(image, mask=None)
        image2, _ = preprocess_fn(image, mask=None)

        image = tf.concat([image1, image2], -1)
        return image

    train_ds = tf.data.Dataset.from_tensor_slices(train_list)
    train_ds = train_ds.shuffle(buffer_size=len(train_list)).repeat()
    train_ds = train_ds.map(
        lambda x: read_dicom_file(x, True), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(
        batch_size=batch_size, drop_remainder=drop_remainder)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_list)
    val_ds = val_ds.map(
        lambda x: read_dicom_file(x, False), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


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

            image_raw = image.tobytes()
            patientid = tfio.image.decode_dicom_data(image_bytes, tags=tfio.image.dicom_tags.PatientID) 
            
            feature = {
                'image_raw': _bytes_feature(image_raw),
                'patient_id': _bytes_feature(patientid)

            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            current_time = time.time()
            elapsed_time = current_time - start_time
            approx_total_time = elapsed_time * (num_files / (idx + 1))
            remaining_time = int(approx_total_time - elapsed_time)

            if idx % 100 == 0:
                print(f'{idx+1} out of {num_files} images have been processed.')
                print('remaining time: {}'.format(str(datetime.timedelta(seconds=remaining_time))))

if __name__ == '__main__':

    convert_dicom_to_tfrecords("dicom_files.txt", "00m", "01-of-09.tfrecords")\