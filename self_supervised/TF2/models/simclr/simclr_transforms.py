import tensorflow as tf
from functools import partial
from self_supervised.TF2.models.simclr.simclr_flags import FLAGS

def random_apply(func, p, x):

    image = tf.cond(tf.less(
                    tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
                    lambda: func(x),
                    lambda: x)

    return image

def random_brightness(image, max_delta, impl='v1'):

    if impl == 'v2':
        factor = tf.random.uniform([], tf.math.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
        image = image * factor
    elif impl == 'v1':
        image = tf.image.random_brightness(image, max_delta=max_delta)

    return image

def color_jitter(image, strength=1.0, random_order=True, impl='v1'):

    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength

    if random_order:
        return color_jitter_rand(image, brightness, contrast, saturation, hue, impl=impl)
    else:
        return color_jitter_nonrand(image, brightness, contrast, saturation, hue, impl=impl)

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
            x = random_brightness(x, max_delta=brightness, impl=impl)
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
        image = apply_transform(i, image, brightness, contrast, saturation, hue)
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
                return random_brightness(x, max_delta=brightness, impl=impl)

        def contrast_foo():
            if contrast == 0:
                return x
            else:
                return tf.image.random_contrast(x, lower=1 - contrast, upper=1 + contrast)

        def saturation_foo():
            if saturation == 0:
                return x
            else:
                return tf.image.random_saturation(x, lower=1 - saturation, upper=1 + saturation)

        def hue_foo():
            if hue == 0:
                return x
            else:
                return tf.image.random_hue(x, max_delta=hue)

        x = tf.cond(tf.less(i, 2),
                    lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                    lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
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

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(image_size=image.shape,
                                                                           bounding_boxes=bbox,
                                                                           min_object_covered=min_object_covered,
                                                                           aspect_ratio_range=aspect_ratio_range,
                                                                           area_range=area_range,
                                                                           max_attempts=max_attempts,
                                                                           use_image_if_no_bounding_boxes=True)

    bbox_start, bbox_size, _ = sample_distorted_bounding_box

    offset_y, offset_x, _ = tf.unstack(bbox_start)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)

    return image

def center_crop(image, image_size, crop_proportion):

    image = tf.image.central_crop(image, crop_proportion)
    image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)

    return image

def crop_and_resize(image, image_size):

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = image_size[0] / image_size[1]
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.08, 1.0),
        max_attempts=100)
    image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)

    return image

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.math.exp(
        -tf.math.pow(x, 2.0) / (2.0 * tf.math.pow(tf.cast(sigma, tf.float32), 2.0))
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

    blurred = tf.nn.depthwise_conv2d(image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)

    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)

    return blurred

def random_crop_with_resize(image, image_size, p=1.0):

    def _transform(image):
        image = crop_and_resize(image, image_size)
        return image
    return random_apply(_transform, p=p, x=image)

def random_color_jitter(image, strength, p=1.0):

    def _transform(image):
        color_jitter_t = partial(color_jitter, strength=strength)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(color_drop, p=0.2, x=image)
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

def preprocess_for_train(image,
                         image_size,
                         color_distort=True,
                         crop=True,
                         flip=True):

    if crop:
        image = random_crop_with_resize(image, image_size)

    if flip:
        image = tf.image.random_flip_left_right(image)

    if color_distort:
        image = random_color_jitter(image, strength=0.5)

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

def preprocess_image(image, image_size, is_training=False, color_distort=True, test_crop=True):

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, image_size, color_distort)
    else:
        return preprocess_for_eval(image, image_size, test_crop)

def get_preprocess_fn(is_training, is_pretrain):

    if FLAGS.dataset == 'cifar10':
        test_crop = False
    else:
        test_crop = True

    return partial(preprocess_image,
                   image_size=[FLAGS.image_size, FLAGS.image_size],
                   is_training=is_training,
                   color_distort=is_pretrain,
                   test_crop=test_crop)
