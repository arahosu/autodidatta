import tensorflow as tf
from tensorflow.keras.layers import Layer, Wrapper


class BaseOps(Layer):

    def __init__(self,
                 p=1.0,
                 seed=None,
                 name=None,
                 trainable=False,
                 dtype=None,
                 dynamic=False,
                 **kwargs):

        super(BaseOps, self).__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs
            )

        self.p = p
        self.seed = seed

    def call(self, inputs, training=None):
        return inputs

    def apply(self, image, seg=None, training=True):
        return image, seg


class ImageOnlyOps(BaseOps):

    def apply(self, image, seg=None, training=True):

        cond = tf.less(tf.random.uniform([], seed=self.seed), self.p)
        outputs = tf.cond(
            cond,
            lambda: self(image, training=training),
            lambda: image
        )

        return outputs, seg


class DualOps(BaseOps):

    def apply(self, image, seg=None, training=True):
        cond = tf.less(
            tf.random.uniform([], seed=self.seed), self.p)
        if seg is not None:
            new_image = tf.concat([image, seg], -1)
            outputs = tf.cond(
                cond,
                lambda: self(new_image, training=training),
                lambda: new_image
                )
            image = outputs[..., :image.shape[-1]]
            seg = outputs[..., image.shape[-1]:]
            return image, seg
        else:
            image = tf.cond(
                cond,
                lambda: self(image, training=training),
                lambda: image
                )
            return image, seg


class ImageOnlyOpsWrapper(Wrapper):
    def __init__(self,
                 layer,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):

        super(ImageOnlyOpsWrapper, self).__init__(
            layer=layer,
            name=name,
            **kwargs
            )

        self.layer = layer
        self.p = p
        self.seed = seed

    def apply(self, image, seg=None, training=True):

        cond = tf.less(tf.random.uniform([], seed=self.seed), self.p)
        outputs = tf.cond(
            cond,
            lambda: self.layer(image, training=training),
            lambda: image
        )

        return outputs, seg


class DualOpsWrapper(Wrapper):
    def __init__(self,
                 layer,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):

        super(DualOpsWrapper, self).__init__(
            layer=layer,
            name=name,
            **kwargs
            )

        self.layer = layer
        self.p = p
        self.seed = seed

    def apply(self, image, seg=None, training=True):

        cond = tf.less(
            tf.random.uniform([], seed=self.seed), self.p)
        if seg is not None:
            new_image = tf.concat([image, seg], -1)
            outputs = tf.cond(
                cond,
                lambda: self.layer(new_image, training=training),
                lambda: new_image
                )
            image = outputs[..., :image.shape[-1]]
            seg = outputs[..., image.shape[-1]:]
            return image, seg
        else:
            image = tf.cond(
                cond,
                lambda: self.layer(image, training=training),
                lambda: image
                )
            return image, seg
