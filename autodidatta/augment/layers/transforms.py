import tensorflow as tf
from autodidatta.augment.layers.base import BaseOps


class RandomBrightness(BaseOps):
    def __init__(self,
                 factor,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):
        super(RandomBrightness, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = 1 - factor
            self.upper = 1 + factor

        if factor < 0:
            raise ValueError('Factor must be non-negative.',
                             ' got {}'.format(factor))

        self.seed = seed

    def apply(self, inputs, training=True):
        image_dtype = inputs.dtype
        delta = tf.random.uniform(
            [], tf.maximum(self.lower, 0),
            tf.maximum(self.upper, 0), seed=self.seed)
        if training:
            inputs = tf.cast(inputs, tf.float32)
            image = inputs * delta
            return tf.cast(image, image_dtype)
        else:
            return inputs


class RandomContrast(BaseOps):

    def __init__(self,
                 factor,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):

        super(RandomContrast, self).__init__(
            p=p, seed=seed, name=name, **kwargs
            )

        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = 1 - factor
            self.upper = 1 + factor

        self.seed = seed

    def apply(self, inputs, training=True):
        if training:
            return tf.image.random_contrast(
                inputs, self.lower, self.upper, self.seed
            )
        else:
            return inputs


class RandomGamma(BaseOps):

    def __init__(self,
                 gamma,
                 gain,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):
        super(RandomGamma, self).__init__(
            p=p, seed=seed, name=name, **kwargs
            )

        if isinstance(gamma, (tuple, list)):
            self.lower_gamma = gamma[0]
            self.upper_gamma = gamma[1]
        else:
            self.lower_gamma = 0.
            self.upper_gamma = gamma
        if isinstance(gain, (tuple, list)):
            self.lower_gain = gain[0]
            self.upper_gain = gain[1]
        else:
            self.lower_gain = gain
            self.upper_gain = gain

        if self.lower_gamma < 0. or self.upper_gamma < 0.:
            raise ValueError('Gamma cannot have negative values',
                             ' got {}'.format(gamma))

        self.seed = seed

    def apply(self, inputs, training=True):

        random_gamma = tf.random.uniform(
            [], self.lower_gamma, self.upper_gamma, seed=self.seed)
        random_gain = tf.random.uniform(
            [], self.lower_gain, self.upper_gain, seed=self.seed)

        if training:
            image = random_gain * tf.math.sign(inputs) * \
                (tf.math.abs(inputs) + 1e-08)**random_gamma
            return image
        else:
            return inputs


class RandomSaturation(BaseOps):

    def __init__(self,
                 factor,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):
        super(RandomSaturation, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = 1 - factor
            self.upper = 1 + factor

        self.seed = seed

    def apply(self, inputs, training=True):
        if training:
            return tf.image.random_saturation(
                inputs, self.lower, self.upper, seed=self.seed
            )
        else:
            return inputs


class RandomHue(BaseOps):

    def __init__(self,
                 factor,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):
        super(RandomHue, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        self.factor = factor
        self.seed = seed

    def apply(self, inputs, training=True):
        if training:
            return tf.image.random_hue(
                inputs, self.factor, seed=self.seed)
        else:
            return inputs


class ColorJitter(BaseOps):

    def __init__(self,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 clip_value=False,
                 random_order=True,
                 p=0.8,
                 seed=None,
                 name=None,
                 **kwargs):
        super(ColorJitter, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        self.brightness_op = RandomBrightness(brightness, seed=seed)
        self.contrast_op = RandomContrast(contrast, seed=seed)
        self.saturation_op = RandomSaturation(saturation, seed=seed)
        self.hue_op = RandomHue(hue, seed=seed)

        self.clip_value = clip_value
        self.random_order = random_order
        self.seed = seed

    def apply(self, inputs, training=True):
        image_dtype = inputs.dtype
        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            idx = perm[i] if self.random_order else i
            inputs = self.apply_transform(
                inputs, idx, training=training)
            if self.clip_value:
                inputs = tf.clip_by_value(inputs, 0, 1)
        inputs = tf.cast(inputs, image_dtype)
        return inputs

    def apply_transform(self, inputs, i, training=True):

        if i == 0:
            inputs = self.brightness_op.apply(inputs, training=training)
        elif i == 1:
            inputs = self.contrast_op.apply(inputs, training=training)
        elif i == 2:
            inputs = self.saturation_op.apply(inputs, training=training)
        elif i == 3:
            inputs = self.hue_op.apply(inputs, training=training)

        return inputs


class Solarize(BaseOps):

    def __init__(self,
                 threshold=127,
                 p=0.2,
                 seed=None,
                 name=None,
                 **kwargs):
        super(Solarize, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        self.threshold = threshold
        self.seed = seed

    def apply(self, inputs, training=True):
        if self.threshold < 1.0 and inputs.dtype != tf.uint8:
            maxval = 1.0
        else:
            maxval = 255
        
        assert self.threshold < maxval, 'threshold cannot be greater than the maximum value'

        if training:
            image = tf.where(inputs < self.threshold, inputs, maxval - inputs)
            return tf.clip_by_value(image, 0, maxval)
        else:
            return inputs


class ToGray(BaseOps):
    def __init__(self, p=0.2, name=None, **kwargs):
        super(ToGray, self).__init__(
            p=p, seed=None, name=name, **kwargs
        )

    def apply(self, inputs, training=True):
        image_dtype = inputs.dtype
        if training:
            image = tf.image.rgb_to_grayscale(inputs)
            image = tf.tile(image, [1, 1, 3])
            image = tf.cast(image, image_dtype)
            return image
        else:
            return inputs


class GaussianBlur(BaseOps):
    #TODO: Simpler implementation of GaussianBlur
    def __init__(self,
                 kernel_size,
                 sigma,
                 padding='SAME',
                 p=0.5,
                 name=None,
                 **kwargs):
        
        super(GaussianBlur, self).__init__(
            p=p, seed=None, name=name, **kwargs
        )

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = padding
        
    def apply(self, inputs, training=True):
        image_dtype = inputs.dtype
        if training:
            inputs = tf.cast(inputs, tf.float32)
            radius = tf.cast(self.kernel_size / 2, dtype=tf.int32)
            kernel_size = radius * 2 + 1
            x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
            blur_filter = tf.exp(-tf.pow(x, 2.0) /
                                (2.0 * tf.pow(tf.cast(self.sigma, dtype=tf.float32), 2.0)))
            blur_filter /= tf.reduce_sum(blur_filter)

            # One vertical and one horizontal filter.
            blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
            blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
            num_channels = tf.shape(inputs)[-1]
            blur_h = tf.cast(tf.tile(blur_h, [1, 1, num_channels, 1]), tf.float32)
            blur_v = tf.cast(tf.tile(blur_v, [1, 1, num_channels, 1]), tf.float32)
            expand_batch_dim = inputs.shape.ndims == 3
            if expand_batch_dim:
                # Tensorflow requires batched input to convolutions, which we can fake with
                # an extra dimension.
                image = tf.expand_dims(inputs, axis=0)
            blurred = tf.nn.depthwise_conv2d(
                image, blur_h, strides=[1, 1, 1, 1], padding=self.padding)
            blurred = tf.nn.depthwise_conv2d(
                blurred, blur_v, strides=[1, 1, 1, 1], padding=self.padding)
            if expand_batch_dim:
                blurred = tf.squeeze(blurred, axis=0)
            blurred = tf.cast(blurred, image_dtype)
            return blurred
        else:
            return inputs


class Normalize(BaseOps):
    def __init__(self,
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.247, 0.243, 0.261],
                 rescale=True,
                 name=None,
                 **kwargs):
        super(Normalize, self).__init__(
            p=1.0, always_apply=True, seed=None, name=name, **kwargs
        )

        self.mean = mean
        self.std = std
        self.rescale = rescale

    def apply(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)
        if self.rescale:
            inputs /= 255.
        # The function is always called by default
        image = (inputs - self.mean) / self.std
        return image
