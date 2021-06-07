import tensorflow as tf
import os
from functools import partial

from sss.augmentation.base import jigsaw, random_crop_with_resize
from sss.augmentation.dual_transform import get_preprocess_fn
AUTOTUNE = tf.data.AUTOTUNE


def normalize_image(image_tensor):
    mean = tf.math.reduce_mean(image_tensor)
    std = tf.math.reduce_std(image_tensor)
    not_blank = std > 0.0

    image_tensor = tf.cond(not_blank,
                           true_fn=lambda: tf.divide(
                               tf.math.subtract(image_tensor, mean), std),
                           false_fn=lambda: image_tensor)

    return image_tensor


def background(image):

    image_not_background = tf.clip_by_value(
        tf.math.reduce_sum(image, axis=-1), 0, 1)
    image_background = tf.expand_dims(
        tf.math.logical_not(
            tf.cast(image_not_background, dtype=tf.bool)), axis=-1)
    image_background = tf.cast(image_background, dtype=tf.float32)
    return tf.concat([image_background, image], axis=-1)


def parse_fn_2d(example_proto,
                training_mode,
                is_training,
                multi_class,
                add_background,
                normalize):

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
        image = normalize_image(image)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int8)
    seg = tf.reshape(seg_raw, [384, 384, 6])
    if not multi_class:
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, -1)
    seg = tf.cast(seg, tf.float32)
    seg = tf.clip_by_value(seg, 0., 1.)

    tf.debugging.check_numerics(image, "Invalid value in your input!")
    tf.debugging.check_numerics(seg, "Invalid value in your label!")

    preprocess_fn_pretrain = get_preprocess_fn(
        is_training=is_training, is_pretrain=True, image_size=288)
    preprocess_fn_finetune = get_preprocess_fn(
        is_training=is_training, is_pretrain=False, image_size=288)

    if training_mode == 'pretrain':
        image1, seg1 = preprocess_fn_pretrain(
            image=image, mask=seg)
        image2, seg2 = preprocess_fn_pretrain(
            image=image, mask=seg)

        image = tf.concat([image1, image2], -1)
        if multi_class:
            if add_background:
                seg1, seg2 = background(seg1), background(seg2)
        seg = tf.concat([seg1, seg2], -1)
        return (image, seg)
    else:
        image, seg = preprocess_fn_finetune(
            image=image, mask=seg)
        if multi_class:
            if add_background:
                seg = background(seg)
        return (image, seg)


def parse_fn_rotate(example_proto,
                    is_training,
                    multi_class,
                    add_background,
                    normalize):

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
    if not multi_class:
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, -1)
    seg = tf.cast(seg, tf.float32)
    seg = tf.clip_by_value(seg, 0., 1.)

    tf.debugging.check_numerics(image, "Invalid value in your input!")
    tf.debugging.check_numerics(seg, "Invalid value in your label!")

    preprocess_fn_finetune = get_preprocess_fn(
        is_training=is_training, is_pretrain=False, image_size=288)

    image, seg = preprocess_fn_finetune(
            image=image, mask=seg)
    num_rot = tf.random.uniform([], 1, 5, dtype=tf.int32)
    if is_training:
        image = tf.image.rot90(image, k=num_rot)
        seg = tf.image.rot90(seg, k=num_rot)

    if normalize:
        image = normalize_image(image)

    if multi_class:
        if add_background:
            seg = background(seg)

    num_rot = tf.cast(num_rot, tf.float32) - 1

    return (image, seg, num_rot)


def parse_fn_restore(example_proto,
                     is_training,
                     multi_class,
                     add_background,
                     normalize,
                     permutations):

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
    if not multi_class:
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, -1)
    seg = tf.cast(seg, tf.float32)
    seg = tf.clip_by_value(seg, 0., 1.)

    tf.debugging.check_numerics(image, "Invalid value in your input!")
    tf.debugging.check_numerics(seg, "Invalid value in your label!")

    # preprocess_fn_finetune = get_preprocess_fn(
    #     is_training=is_training, is_pretrain=False, image_size=288)

    # image, seg = preprocess_fn_finetune(
    #         image=image, mask=seg)

    if normalize:
        image = normalize_image(image)

    combined_img = tf.concat([image, seg], axis=-1)
    if is_training:
        combined_img_cropped = random_crop_with_resize(
            combined_img, [288, 288], area_range=(0.5625, 1.0))
        combined_img_cropped = tf.reshape(
            combined_img_cropped, [288, 288, combined_img.shape[-1]])
        combined_image_shuffle, label = jigsaw(
            combined_img_cropped, permutations)
        image = combined_image_shuffle[..., 0]
        seg = combined_image_shuffle[..., 1:]
        image = tf.expand_dims(image, -1)
        image_shuffle = image
    else:
        image = tf.image.resize_with_crop_or_pad(
            image, 288, 288)
        seg = tf.image.resize_with_crop_or_pad(
            seg, 288, 288)
        image_shuffle, label = jigsaw(image, permutations)

    if multi_class:
        if add_background:
            seg = background(seg)

    return (image, seg, image_shuffle, label)


def parse_fn_viz(example_proto,
                 training_mode,
                 is_training,
                 multi_class,
                 add_background,
                 normalize):

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
        image = normalize_image(image)

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int8)
    seg = tf.reshape(seg_raw, [384, 384, 6])
    if not multi_class:
        seg = tf.math.reduce_sum(seg, axis=-1)
        seg = tf.expand_dims(seg, -1)
    seg = tf.cast(seg, tf.float32)
    seg = tf.clip_by_value(seg, 0., 1.)

    tf.debugging.check_numerics(image, "Invalid value in your input!")
    tf.debugging.check_numerics(seg, "Invalid value in your label!")

    preprocess_fn_pretrain = get_preprocess_fn(
        is_training=is_training, is_pretrain=True, image_size=288)
    preprocess_fn_finetune = get_preprocess_fn(
        is_training=is_training, is_pretrain=False, image_size=288)

    if training_mode == 'pretrain':
        image1, seg1 = preprocess_fn_pretrain(
            image=image, mask=seg)
        image2, seg2 = preprocess_fn_pretrain(
            image=image, mask=seg)

        image_concat = tf.concat([image1, image2], -1)
        if multi_class:
            if add_background:
                seg1, seg2 = background(seg1), background(seg2)
        seg_concat = tf.concat([seg1, seg2], -1)
        return (image, seg, image_concat, seg_concat)
    else:
        image, seg = preprocess_fn_finetune(
            image=image, mask=seg)
        if multi_class:
            if add_background:
                seg = background(seg)
        return (image, seg)


def read_tfrecord(tfrecords_dir,
                  batch_size,
                  buffer_size,
                  is_training,
                  training_mode,
                  fraction_data,
                  multi_class,
                  add_background,
                  normalize,
                  permutations=None,
                  parse_fn=parse_fn_2d):

    """This function reads and returns TFRecords dataset in tf.data.Dataset format
    """

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

    if parse_fn == parse_fn_2d:
        dataset = dataset.map(
            partial(
                parse_fn, training_mode=training_mode,
                is_training=is_training,
                multi_class=multi_class,
                add_background=add_background,
                normalize=normalize),
            num_parallel_calls=AUTOTUNE)
    elif parse_fn == parse_fn_rotate:
        dataset = dataset.map(
            partial(
                parse_fn,
                is_training=is_training,
                multi_class=multi_class,
                add_background=add_background,
                normalize=normalize),
            num_parallel_calls=AUTOTUNE)
    elif parse_fn == parse_fn_restore:
        assert permutations is not None, 'Permutations cannot be None'
        dataset = dataset.map(
            partial(
                parse_fn,
                is_training=is_training,
                multi_class=multi_class,
                add_background=add_background,
                normalize=normalize,
                permutations=permutations),
            num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(
            partial(
                parse_fn, training_mode=training_mode,
                is_training=is_training,
                multi_class=multi_class,
                add_background=add_background,
                normalize=normalize),
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
                 fraction_data,
                 multi_class,
                 add_background,
                 normalize,
                 permutations=None,
                 buffer_size=19200,
                 parse_fn=parse_fn_2d):

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
        training_mode=training_mode,
        fraction_data=fraction_data,
        multi_class=multi_class,
        add_background=add_background,
        normalize=normalize,
        permutations=permutations,
        parse_fn=parse_fn
        )

    valid_ds = read_tfrecord(
        tfrecords_dir=os.path.join(dataset_dir, valid_dir),
        batch_size=batch_size,
        buffer_size=buffer_size,
        is_training=False,
        training_mode=training_mode,
        fraction_data=1.0,
        multi_class=multi_class,
        add_background=add_background,
        normalize=normalize,
        permutations=permutations,
        parse_fn=parse_fn
        )

    return train_ds, valid_ds
