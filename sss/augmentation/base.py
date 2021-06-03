import tensorflow as tf
import itertools
import numpy as np


def random_apply(func, p, x):

    image = tf.cond(tf.less(
                    tf.random.uniform(
                        [], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
                    lambda: func(x),
                    lambda: x)

    return image


def random_brightness(image,
                      max_delta,
                      impl='v1'):

    if impl == 'v2':
        factor = tf.random.uniform(
            [], tf.math.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
        image = image * factor
    elif impl == 'v1':
        image = tf.image.random_brightness(image, max_delta=max_delta)

    return image


def random_gamma(image, max_gamma):
    gamma = tf.random.uniform(shape=[], minval=0.5, maxval=max_gamma)
    return tf.math.sign(image) * (tf.math.abs(image) + 1e-08)**gamma


def random_gaussian_noise(image, noise):
    noise = tf.random.normal(image.shape, mean=0.0, stddev=noise)
    image += noise
    return image


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        image_size=image.shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)

    bbox_start, bbox_size, _ = sample_distorted_bounding_box

    offset_y, offset_x, _ = tf.unstack(bbox_start)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, target_height, target_width)

    return image


def center_crop(image, image_size, crop_proportion):

    image = tf.image.central_crop(image, crop_proportion)
    image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)

    return image


def crop_and_resize(image, image_size, area_range=(0.08, 1.0)):

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = image_size[0] / image_size[1]

    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=area_range,
        max_attempts=100)
    image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)

    return image


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.math.exp(
        -tf.math.pow(x, 2.0) / (2.0 * tf.math.pow(
            tf.cast(sigma, tf.float32), 2.0))
    )
    blur_filter /= tf.math.reduce_sum(blur_filter)

    # one vertical and one horizontal filter
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = image.shape[-1]

    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3

    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)

    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)

    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)

    return blurred


def random_crop_with_resize(image,
                            image_size,
                            area_range=(0.08, 1.0),
                            p=1.0):

    def _transform(image):
        image = crop_and_resize(
            image, image_size, area_range=area_range)
        return image
    return random_apply(_transform, p=p, x=image)


def random_blur(image, image_size, p=1.0):

    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(
            image,
            kernel_size=image_size[1] // 10,
            sigma=sigma,
            padding='SAME'
        )

    return random_apply(_transform, p=p, x=image)


def jigsaw(image, patches_per_side=3, num_labels=64):
    M = int(image.shape[0] // patches_per_side)
    N = int(image.shape[1] // patches_per_side)
    patch_list = []
    shuffle_image = tf.zeros_like(image)

    for x in range(0, image.shape[0], M):
        for y in range(0, image.shape[1], N):
            patch = image[x:x+M, y:y+N, :]
            patch_list.append(patch)

    indices = [i for i in range(patches_per_side*patches_per_side)]
    perm = list(itertools.permutations(indices))
    new_list = [list(perm[i]) for i in range(len(perm))]
    if num_labels <= len(perm):
        new_list = new_list[-num_labels:]
    # rand_label_idx = tf.random.uniform([], 1, len(new_list), tf.int32)
    rand_label_idx = np.random.randint(0, len(new_list))
    patch_list = [patch_list[i] for i in new_list[rand_label_idx]]

    idx = 0
    for x in range(0, image.shape[0], M):
        for y in range(0, image.shape[1], N):
            update_elem = patch_list[idx]
            update_elem = tf.reshape(update_elem, [-1, image.shape[-1]])
            update_indices = [[a, b] for a in range(x, x+M) for b in range(y, y + N)]
            a = tf.tensor_scatter_nd_update(
                shuffle_image, update_indices, update_elem)
            shuffle_image = a
            idx += 1

    # rand_label_idx += 1
    rand_label_idx = tf.cast(rand_label_idx, tf.float32)

    return shuffle_image, rand_label_idx
