import tensorflow as tf
from tensorflow.keras.layers import Layer
from functools import partial

class BaseOps(Layer):
    """ A layer from which all autodidatta.augment.layers inherit
        
    Args:
    p: float, probability of applying the transform
    always_apply: bool, Set this to 'True' if your augmentation 
        function should be applied during inference. This argument does 
        not apply if your augmentation layers are defined in the datasets,
        and not inside the model.
    seed: Random seed. Must have dtype int32 or int64 
        (When using XLA/TPU, only int32 is allowed)
    name: str, a name for the operation (optional)
    """

    def __init__(self,
                 p=1.0,
                 always_apply=False,
                 name=None,
                 **kwargs):

        super(BaseOps, self).__init__(
            name=name,
            **kwargs
            )

        # if seed is None:
        #     seed = tf.random.uniform((2,), maxval=int(1e10), dtype=tf.int32)

        self.p = p
        # self.seed = seed
        self.always_apply = always_apply

    def call(self, inputs, training=True):

        # If a 3-D tensor of shape [height, width, depth]
        if inputs.shape.ndims == 3:
            outputs = self.apply_fn(inputs, training)
        # If a 4-D tensor of shape [batch size, height, width, depth]
        elif inputs.shape.ndims == 4:
            output_fn = partial(self.apply_fn, training=training)
            outputs = tf.map_fn(output_fn, inputs)
        else:
            raise ValueError('Inputs must be a 3D or 4D tensor. \
            Found ndims == {}'.format(inputs.shape.ndims))
        return outputs

    def apply(self, inputs, training=True):
        raise NotImplementedError("apply is not implemented in BaseOps")
        
    def apply_fn(self, inputs, training=True):
        cond = tf.less(tf.random.uniform([]), self.p)
        if not self.always_apply:
            outputs = tf.cond(
                cond,
                lambda: self.apply(inputs, training=training),
                lambda: inputs
            )
        else:
            outputs = self.apply(inputs, training=True)
        return outputs
        
        