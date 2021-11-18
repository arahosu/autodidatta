import tensorflow as tf
from tensorflow.keras.layers import Layer, Wrapper
from functools import partial

class BaseOps(Layer):

    def __init__(self,
                 p=1.0,
                 always_apply=False,
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
        self.always_apply = always_apply

    def call(self, inputs, training=True):
        if inputs.shape.ndims == 3:
            outputs = self.apply_fn(inputs, training)
        elif image.shape.ndims == 4:
            output_fn = partial(self.apply_fn, training=training)
            outputs = tf.map_fn(output_fn, inputs)
        return outputs

    def apply(self, image, training=True):
        raise NotImplementedError("apply is not implemented in BaseOps")
        
    def apply_fn(self, image, training=True):
        cond = tf.less(tf.random.uniform([], seed=self.seed), self.p)
        if not self.always_apply:
            outputs = tf.cond(
                cond,
                lambda: self.apply(image, training=training),
                lambda: image
            )
        else:
            outputs = self.apply(image, training=True)
        return outputs