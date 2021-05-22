import tensorflow as tf
from functools import partial

from sss.augmentation.base import random_apply, \
    random_brightness, random_gamma, random_gaussian_noise, \
    random_crop_with_resize, center_crop, random_blur
from sss.flags import FLAGS


def mono_jitter_rand(image,
                     brightness,
                     contrast,
                     gamma,
                     noise,
                     impl='v1'):

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

        def gamma_foo():
            if gamma == 0:
                return x
            else:
                return random_gamma(image, gamma)

        def noise_foo():
            if noise == 0:
                return x
            else:
                return random_gaussian_noise(image, noise)

        x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(
                        tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(
                        tf.less(i, 3), gamma_foo, noise_foo))
        return x

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
        image = apply_transform(perm[i], image)
        image = tf.clip_by_value(image, 0., 1.)
    return image


def mono_jitter(image, strength=1.0, impl='v1'):

    brightness = 0.4 * strength
    contrast = 0.4 * strength
    gamma = 0.2 * strength
    noise = 1.0 * strength

    return mono_jitter_rand(image, brightness, contrast, gamma, noise)


def random_mono_jitter(image, strength, p=1.0):

    def _transform(image):
        mono_jitter_t = partial(mono_jitter, strength=strength)
        return mono_jitter_t(image)
    return random_apply(_transform, p=p, x=image)


def preprocess_for_train(image,
                         image_size,
                         mask=None,
                         distort=True,
                         crop=True,
                         flip=True):

    if distort:
        image = random_mono_jitter(image, strength=0.5)

    image = random_blur(image, image_size, p=0.5)

    if mask is not None:
        mask_shape = mask.shape
        num_image_ch = image.shape[-1]
        image = tf.concat([image, mask], axis=-1)

    if crop:
        image = random_crop_with_resize(image, image_size)
    else:
        if FLAGS.dataset == 'oai':
            crop_factor = 0.5625
        else:
            crop_factor = 0.875
        image = center_crop(image, image_size, crop_factor)

    if flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.clip_by_value(image, 0., 1.)

    if mask is not None:
        new_image = image[..., :num_image_ch]
        mask = image[..., num_image_ch:]

        new_image = tf.reshape(
            new_image, [image_size[0], image_size[1], num_image_ch])
        mask = tf.reshape(mask, [image_size[0], image_size[1], mask_shape[-1]])

    return new_image, mask


def preprocess_for_eval(image,
                        image_size,
                        mask=None,
                        crop=True):

    if mask is not None:
        num_image_ch = image.shape[-1]
        mask_shape = mask.shape
        image = tf.concat([image, mask], axis=-1)

    if crop:
        image = center_crop(image, image_size, 0.5625)

    image = tf.clip_by_value(image, 0., 1.)

    if mask is not None:
        new_image = image[..., :num_image_ch]
        mask = image[..., num_image_ch:]

        new_image = tf.reshape(
            new_image, [image_size[0], image_size[1], num_image_ch])
        mask = tf.reshape(
            mask, [image_size[0], image_size[1], mask_shape[-1]])

    return new_image, mask


def preprocess_image(image,
                     image_size,
                     mask,
                     is_training=False,
                     distort=True,
                     test_crop=True):

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, image_size, mask, distort)
    else:
        return preprocess_for_eval(image, image_size, mask, test_crop)


def get_preprocess_fn(is_training, is_pretrain):

    return partial(preprocess_image,
                   image_size=[FLAGS.image_size, FLAGS.image_size],
                   is_training=is_training,
                   distort=is_pretrain,
                   test_crop=True)
