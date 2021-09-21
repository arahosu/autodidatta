from functools import partial
import os
import tensorflow as tf
import autodidatta.augment as A
AUTOTUNE = tf.data.AUTOTUNE


def min_max_normalize(image):
    denom = tf.math.reduce_max(image) - tf.math.reduce_min(image)
    num = image - tf.math.reduce_min(image)

    return num / denom


def add_background_class(image):

    image_not_background = tf.clip_by_value(
        tf.math.reduce_sum(image, axis=-1), 0, 1)
    image_background = tf.expand_dims(
        tf.math.logical_not(
            tf.cast(image_not_background, dtype=tf.bool)), axis=-1)
    image_background = tf.cast(image_background, dtype=tf.float32)
    return tf.concat([image_background, image], axis=-1)


def aug_fn(image, image_size, is_training, pre_train, mask=None):

    if not pre_train:
        assert mask is not None, \
            'mask must be specified and not be None if pre_train is False'

    jitter_prob = 0.8 if pre_train else 0.

    transforms = A.Augment([
        A.layers.RandomResizedCrop(image_size, image_size, scale=(0.65, 1.0)),
        A.layers.RandomBrightness(0.4, p=jitter_prob),
        A.layers.RandomContrast(0.4, p=jitter_prob),
        A.layers.RandomGamma(gamma=0.5, gain=1, p=jitter_prob)]
    )

    # apply augmentation
    if mask is None:
        aug_img = transforms(image, training=is_training)
        return aug_img
    else:
        aug_img, aug_mask = transforms(image, seg=mask, training=is_training)
        return aug_img, aug_mask


def parse_fn_pretrain(example_proto,
                      is_training,
                      image_size,
                      normalize,
                      patient_id_exclusion_list=None):

    features = {
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'patient_id': tf.io.FixedLenFeature([], tf.int64)
                }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.int16)
    image = tf.cast(tf.reshape(image_raw, [384, 384, 1]), tf.float32)

    if normalize:
        image = min_max_normalize(image)

    patient_id = image_features['patient_id']

    if patient_id_exclusion_list is not None:
        if patient_id not in patient_id_exclusion_list:
            preprocess_fn = partial(
                aug_fn, image_size=image_size,
                is_training=is_training, pre_train=True)

            image_1 = preprocess_fn(image)
            image_2 = preprocess_fn(image)
            if not is_training:
                image_1 = tf.image.resize_with_crop_or_pad(
                    image_1, image_size, image_size)
                image_2 = tf.image.resize_with_crop_or_pad(
                    image_2, image_size, image_size)
            image_1 = tf.reshape(image_1, [image_size, image_size, 1])
            image_2 = tf.reshape(image_2, [image_size, image_size, 1])
            image = tf.concat([image_1, image_2], -1)

            return image
    else:
        preprocess_fn = partial(
            aug_fn, image_size=image_size,
            is_training=is_training, pre_train=True)

        image_1 = preprocess_fn(image)
        image_2 = preprocess_fn(image)
        if not is_training:
            image_1 = tf.image.resize_with_crop_or_pad(
                image_1, image_size, image_size)
            image_2 = tf.image.resize_with_crop_or_pad(
                image_2, image_size, image_size)
        image_1 = tf.reshape(image_1, [image_size, image_size, 1])
        image_2 = tf.reshape(image_2, [image_size, image_size, 1])
        image = tf.concat([image_1, image_2], -1)

        return image


def parse_fn_finetune(example_proto,
                      is_training,
                      image_size,
                      normalize,
                      multi_class,
                      add_background):

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

    if normalize:
        image = min_max_normalize(image)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int8)
    seg = tf.reshape(seg_raw, [384, 384, 6])

    if not multi_class:
        assert add_background is False, 'if not multi_class, \
            add_background must be set to False'
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, -1)

    if add_background:
        seg = add_background_class(seg)

    seg = tf.cast(seg, tf.float32)
    seg = tf.clip_by_value(seg, 0., 1.)
    num_classes = seg.shape[-1]

    preprocess_fn = partial(
        aug_fn,
        image_size=image_size, is_training=is_training, pre_train=False)

    image, seg = preprocess_fn(
            image=image, mask=seg)
    image = tf.reshape(image, [image_size, image_size, 1])
    seg = tf.reshape(seg, [image_size, image_size, num_classes])
    return (image, seg)


def read_tfrecord_pretrain(tfrecords_dir,
                           batch_size,
                           image_size,
                           buffer_size,
                           is_training,
                           normalize,
                           patient_id_exclusion_list=None):

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
            parse_fn_pretrain,
            is_training=is_training,
            image_size=image_size,
            normalize=normalize,
            patient_id_exclusion_list=patient_id_exclusion_list),
        num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    return dataset


def read_tfrecord_finetune(tfrecords_dir,
                           batch_size,
                           image_size,
                           buffer_size,
                           is_training,
                           fraction_data,
                           multi_class,
                           add_background,
                           normalize):

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    if is_training:
        num_files = int(len(file_list) * fraction_data)
        file_list = file_list[:num_files]
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
                parse_fn_finetune,
                is_training=is_training,
                image_size=image_size,
                normalize=normalize,
                multi_class=multi_class,
                add_background=add_background),
            num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    return dataset


def load_dataset(dataset_dir,
                 batch_size,
                 image_size,
                 buffer_size,
                 training_mode,
                 fraction_data,
                 multi_class,
                 add_background,
                 normalize,
                 patient_id_exclusion_list=None):

    if training_mode == 'finetune':
        train_dir, val_dir = 'train/', 'valid/'

        train_ds = read_tfrecord_finetune(
            tfrecords_dir=os.path.join(dataset_dir, train_dir),
            batch_size=batch_size,
            image_size=image_size,
            buffer_size=buffer_size,
            is_training=True,
            fraction_data=fraction_data,
            multi_class=multi_class,
            add_background=add_background,
            normalize=normalize)

        val_ds = read_tfrecord_finetune(
            tfrecords_dir=os.path.join(dataset_dir, val_dir),
            batch_size=batch_size,
            image_size=image_size,
            buffer_size=buffer_size,
            is_training=False,
            fraction_data=1.0,
            multi_class=multi_class,
            add_background=add_background,
            normalize=normalize)

        return train_ds, val_ds

    elif training_mode == 'pretrain':

        ds = read_tfrecord_pretrain(
            tfrecords_dir=dataset_dir,
            batch_size=batch_size,
            image_size=image_size,
            buffer_size=buffer_size,
            is_training=True,
            normalize=normalize,
            patient_id_exclusion_list=patient_id_exclusion_list)

        return ds


if __name__ == '__main__':
    train_ds = load_dataset(
            'gs://oai-challenge-dataset/data/tfrecords',
            1024,
            288,
            1,
            'pretrain',
            1.0,
            False,
            False,
            True)

    for step, (image) in enumerate(train_ds):
        print(step)