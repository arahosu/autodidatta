import tensorflow as tf
from functools import partial

from sss.augmentation.base import random_apply, \
    random_brightness, random_crop_with_resize, \
    random_blur, center_crop


def color_jitter(image, strength=1.0, random_order=True, impl='v1'):

    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength

    if random_order:
        return color_jitter_rand(
            image, brightness, contrast, saturation, hue, impl=impl)
    else:
        return color_jitter_nonrand(
            image, brightness, contrast, saturation, hue, impl=impl)


def color_jitter_nonrand(image,
                         brightness=0,
                         contrast=0,
                         saturation=0,
                         hue=0,
                         impl='v1'):
    """Distorts the color of the image (jittering order is fixed).
    Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'v1' or 'v2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.
    Returns:
    The distorted image tensor.
    """
    def apply_transform(i, x, brightness, contrast, saturation, hue):
        """Apply the i-th transformation."""
        if brightness != 0 and i == 0:
            x = random_brightness(
                x, max_delta=brightness, impl=impl)
        elif contrast != 0 and i == 1:
            x = tf.image.random_contrast(
                x, lower=1 - contrast, upper=1 + contrast)
        elif saturation != 0 and i == 2:
            x = tf.image.random_saturation(
                x, lower=1 - saturation, upper=1 + saturation)
        elif hue != 0:
            x = tf.image.random_hue(x, max_delta=hue)
        return x

    for i in range(4):
        image = apply_transform(
            i, image, brightness, contrast, saturation, hue)
        image = tf.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0,
                      impl='v1'):
    """Distorts the color of the image (jittering order is random).
    Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
    impl: 'v1' or 'v2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.
    Returns:
    The distorted image tensor.
    """
    def apply_transform(i, x):
        """Apply the i-th transformation."""
        def brightness_foo():
            if brightness == 0:
                return x
            else:
                return random_brightness(
                    x, max_delta=brightness, impl=impl)

        def contrast_foo():
            if contrast == 0:
                return x
            else:
                return tf.image.random_contrast(
                    x, lower=1 - contrast, upper=1 + contrast)

        def saturation_foo():
            if saturation == 0:
                return x
            else:
                return tf.image.random_saturation(
                    x, lower=1 - saturation, upper=1 + saturation)

        def hue_foo():
            if hue == 0:
                return x
            else:
                return tf.image.random_hue(
                    x, max_delta=hue)

        x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(
                        tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(
                        tf.less(i, 3), saturation_foo, hue_foo))
        return x

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
        image = apply_transform(perm[i], image)
        image = tf.clip_by_value(image, 0., 1.)
    return image


def color_drop(image, keep_channels=True):

    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image


def random_color_jitter(image, strength, p=1.0):

    def _transform(image):
        color_jitter_t = partial(color_jitter, strength=strength)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(color_drop, p=0.2, x=image)

    return random_apply(_transform, p=p, x=image)


def preprocess_for_train(image,
                         image_size,
                         color_distort=True,
                         crop=True,
                         flip=True,
                         blur=True):

    if crop:
        image = random_crop_with_resize(image, image_size)

    if flip:
        image = tf.image.random_flip_left_right(image)

    if color_distort:
        image = random_color_jitter(image, strength=0.5)

    if blur:
        image = random_blur(image, image_size, p=0.5)

    height, width = image_size[0], image_size[1]

    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)

    return image


def preprocess_for_eval(image, image_size, crop=True):

    if crop:
        image = center_crop(image, image_size, 0.875)

    height, width = image_size[0], image_size[1]

    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)

    return image


def preprocess_image(image,
                     image_size,
                     is_training=False,
                     color_distort=True,
                     test_crop=True,
                     blur=True):

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(
            image, image_size, color_distort, crop=True, flip=True, blur=blur)
    else:
        return preprocess_for_eval(image, image_size, test_crop)


def get_preprocess_fn(is_training,
                      is_pretrain,
                      image_size,
                      test_crop,
                      blur):

    return partial(preprocess_image,
                   image_size=[image_size, image_size],
                   is_training=is_training,
                   color_distort=is_pretrain,
                   test_crop=test_crop,
                   blur=blur)
