# import volumentations as V
# import albumentations as A

import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial

from self_supervised.TF2.models.simclr.simclr_transforms import random_apply
from self_supervised.TF2.models.simclr.simclr_transforms import distorted_bounding_box_crop
from self_supervised.TF2.models.simclr.simclr_transforms import random_brightness
from self_supervised.TF2.models.simclr.simclr_transforms import center_crop

# NOTE: Albumentation and Volumentation are not compatible with TPUs due to tf.numpy_function
"""
def get_augmentations_3d(patch_size):
    return V.Compose([
        V.RandomResizedCrop(shape=patch_size, scale_limit=(0.65, 1.3)),
        V.Rotate((-15, 15), (-15, 15), (-15, 15), p=0.2),
        V.Flip(0, p=0.5),
        V.Flip(1, p=0.5),
        V.Flip(2, p=0.5),
        V.ElasticTransform(p=0.2),
        V.RandomRotate90(),
        V.GaussianNoise(),
        V.RandomGamma()
    ])

def get_augmentations_2d(patch_size, is_training):
    if is_training:
        return A.Compose([
            A.RandomResizedCrop(height=patch_size[0], width=patch_size[1],scale=(0.65, 1.3)),
            A.Rotate(limit=15, p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(),
            A.RandomRotate90(),
            A.RandomGamma()
        ])
    else:
        return A.Compose([
            A.CenterCrop(height=patch_size[0], width=patch_size[1])
        ])
"""


def crop_and_resize(image, image_size):

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = image_size[0] / image_size[1]

    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.65, 1.0),
        max_attempts=100)

    image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)
    return image


def random_resized_crop(image, image_size, mask=None, p=1.0):
    if mask is not None:
        image = tf.concat([image, mask], axis=-1)
    def _transform(image):
        image = crop_and_resize(image, image_size)
        return image
    return random_apply(_transform, p=p, x=image)


def random_gamma(image, max_gamma):
    gamma = tf.random.uniform(shape=[], minval=0., maxval=max_gamma)
    return tf.image.adjust_gamma(image, gamma, gain=1)


def jitter_rand(image,
                brightness=0,
                contrast=0,
                gamma=0,
                impl='v1'):
    """Distorts the brightness, contrast and gamma of the image (jittering order is random).
    Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for jitter.
    contrast: A float, specifying the contrast for jitter.
    gamma: A float, specifying the gamma for 
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

        def gamma_foo():
            if gamma == 0:
                return x
            else:
                return random_gamma(
                    x, max_gamma=gamma)

        x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(
                        tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(
                        tf.less(i, 3), gamma_foo, gamma_foo
                    ))
        return x

    perm = tf.random.shuffle(tf.range(3))
    for i in range(3):
        image = apply_transform(perm[i], image)
        image = tf.clip_by_value(image, 0., 1.)
    return image


def elastic_transform(image, alpha, sigma):

    rand = tf.random.uniform(
        shape=[image.shape[0], image.shape[1]], minval=0., maxval=1.)
    dx = tfa.image.gaussian_filter2d((rand * 2 - 1),
                                     sigma=sigma,
                                     padding="CONSTANT") * alpha
    dy = tfa.image.gaussian_filter2d((rand * 2 - 1),
                                     sigma=sigma,
                                     padding="CONSTANT") * alpha

    x, y = tf.meshgrid(tf.range(start=0, limit=image.shape[0]),
                       tf.range(start=0, limit=image.shape[1]),
                       indexing='ij')

    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    indices = tf.reshape(x+dx, (-1, 1)), tf.reshape(y+dy, (-1, 1))
    indices = tf.transpose(indices, [2, 1, 0])

    aug_image = tfa.image.resampler(
        tf.cast(tf.expand_dims(image, axis=0), tf.float32), indices)
    aug_image = tf.reshape(aug_image, shape=image.shape)

    return tf.transpose(aug_image, [1, 0, 2])


def jitter(image, strength=1.0, impl='v1'):

    brightness = 0.8 * strength
    contrast = 0.8 * strength
    gamma = 4.0 * strength

    return jitter_rand(
        image, brightness, contrast, gamma, impl=impl)


def apply_random_jitter(image, strength, p=0.8):
    def _transform(image):
        return jitter(image, strength=strength)
    return random_apply(_transform, p=p, x=image)


def preprocess_for_train(image,
                         image_size,
                         mask=None,
                         distort=True,
                         crop=True,
                         flip=True):

    if distort:
        image = apply_random_jitter(image, strength=0.5)

    if mask is not None:
        mask_shape = mask.shape
        num_image_ch = image.shape[-1]
        image = tf.concat([image, mask], axis=-1)

    if crop:
        image = random_resized_crop(image, image_size)

    if flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.clip_by_value(image, 0., 1.)

    if mask is not None:
        new_image = image[..., :num_image_ch]
        mask = image[..., num_image_ch:]

        new_image = tf.reshape(new_image, [image_size[0], image_size[1], num_image_ch])
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

        new_image = tf.reshape(new_image, [image_size[0], image_size[1], num_image_ch])
        mask = tf.reshape(mask, [image_size[0], image_size[1], mask_shape[-1]])

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
                   image_size=[288, 288],
                   is_training=is_training,
                   distort=is_pretrain,
                   test_crop=True)
    
