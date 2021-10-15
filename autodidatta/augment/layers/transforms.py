import tensorflow as tf
from autodidatta.augment.layers.base import ImageOnlyOps


class RandomBrightness(ImageOnlyOps):
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
            self.lower = -factor
            self.upper = factor

        if factor < 0:
            raise ValueError('Factor must be non-negative.',
                             ' got {}'.format(factor))

        self.seed = seed

    def call(self, inputs, training=True):
        delta = tf.random.uniform(
            [], self.lower, self.upper, seed=self.seed)
        if training:
            return tf.image.adjust_brightness(inputs, delta)
        else:
            return inputs


class RandomContrast(ImageOnlyOps):

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

    def call(self, inputs, training=True):
        if training:
            return tf.image.random_contrast(
                inputs, self.lower, self.upper, self.seed
            )
        else:
            return inputs


class RandomGamma(ImageOnlyOps):

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

    def call(self, inputs, training=True):

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


class RandomSaturation(ImageOnlyOps):

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

    def call(self, inputs, training=True):
        if training:
            return tf.image.random_saturation(
                inputs, self.lower, self.upper, seed=self.seed
            )
        else:
            return inputs


class RandomHue(ImageOnlyOps):

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

    def call(self, inputs, training=True):
        if training:
            return tf.image.random_hue(
                inputs, self.factor, seed=self.seed
            )
        else:
            return inputs


class ColorJitter(ImageOnlyOps):

    def __init__(self,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 clip_value=True,
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

    def call(self, inputs, training=True):
        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            idx = perm[i] if self.random_order else i
            inputs = self.apply_transform(
                inputs, idx, training=training)
            if self.clip_value:
                inputs = tf.clip_by_value(inputs, 0., 1.)
        return inputs

    def apply_transform(self, inputs, i, training=True):

        if i == 0:
            inputs = self.brightness_op(inputs, training=training)
        elif i == 1:
            inputs = self.contrast_op(inputs, training=training)
        elif i == 2:
            inputs = self.saturation_op(inputs, training=training)
        elif i == 3:
            inputs = self.hue_op(inputs, training=training)

        return inputs


class Solarize(ImageOnlyOps):

    def __init__(self,
                 threshold=0.5,
                 p=0.2,
                 seed=None,
                 name=None,
                 **kwargs):
        super(Solarize, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        self.threshold = threshold
        self.seed = seed

    def call(self, inputs, training=True):
        if training:
            return tf.where(inputs < threshold, image, 1. - image)
        else:
            return inputs

class ToGray(ImageOnlyOps):
    def __init__(self, p=0.2, name=None, **kwargs):
        super(ToGray, self).__init__(
            p=p, seed=None, name=name, **kwargs
        )

    def call(self, inputs, training=True):
        if training:
            image = tf.image.rgb_to_grayscale(inputs)
            image = tf.tile(image, [1, 1, 3])
            return image
        else:
            return inputs


class Normalize(ImageOnlyOps):
    def __init__(self,
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.247, 0.243, 0.261],
                 rescale=True,
                 name=None,
                 **kwargs):
        super(Normalize, self).__init__(
            p=1.0, seed=None, name=name, **kwargs
        )

        self.mean = mean
        self.std = std
        self.rescale = rescale

    def call(self, inputs, training=None):
        if self.rescale:
            inputs /= 255.
        # The function is always called by default
        image = (inputs - self.mean) / self.std
        return image
