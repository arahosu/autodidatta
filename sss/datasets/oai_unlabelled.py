import tensorflow as tf
import tensorflow_io as tfio

from sss.augmentation.dual_transform import get_preprocess_fn
from sss.utils import min_max_standardize
AUTOTUNE = tf.data.AUTOTUNE


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
        image = image[0, ...]
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


if __name__ == '__main__':

    train_ds, _ = load_oai_full_dataset("dicom_files.txt", 32, 288)

    for image in train_ds:
        print(image.shape)
        break
