from autodidatta.augment.layers.base import BaseOps
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.image import sample_distorted_bounding_box
import matplotlib.pyplot as plt
import math


class RandomResizedCrop(BaseOps):
    """ Inception-style random cropping (Szegedy et al., 2015)
    
    Args:
    height: int, the height of input space.
    width: int, the width of input space.
    scale: [float, float], the range of cropped size.
    ratio: [float, float], the range of cropped aspect ratio.
    interpolation: tf.image.ResizeMethod, flag used to define the 
        interpolation algorithm for resizing the input.
    p: float, probability of applying the transform.
    seed: Random seed. Must have dtype int32 or int64 
        (When using XLA/TPU, only int32 is allowed).
    name: str, a name for the operation (optional).
    """

    def __init__(self,
                 height,
                 width,
                 scale=(0.08, 1.0),
                 ratio=(0.75, 1.33),
                 interpolation=tf.image.ResizeMethod.BICUBIC,
                 p=1.0,
                 seed=None,
                 name=None,
                 **kwargs):

        super(RandomResizedCrop, self).__init__(
            p=p, seed=seed, name=name, **kwargs)

        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _op(self, inputs):
        image_dtype = inputs.dtype
        bbox = tf.constant(
            [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        aspect_ratio = self.height / self.width
        ratio = tuple([aspect_ratio*x for x in self.ratio])

        distorted_bb = sample_distorted_bounding_box(
            image_size=tf.shape(inputs),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=ratio,
            area_range=self.scale,
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)

        bbox_start, bbox_size, _ = distorted_bb

        offset_y, offset_x, _ = tf.unstack(bbox_start)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(
            inputs, offset_y, offset_x, target_height, target_width)

        image = tf.image.resize(
            image, [self.height, self.width], self.interpolation)
        
        image = tf.cast(image, image_dtype)

        return image

    def apply(self, inputs, training=True):
        if training:
            return self._op(inputs)
        else:
            return inputs


class HorizontalFlip(BaseOps):
    """ Applies random horizontal flip

    Args:
    p: float, probability of applying the transform.
    seed: Random seed. Must have dtype int32 or int64 
        (When using XLA/TPU, only int32 is allowed).
    name: str, a name for the operation (optional).
    """

    def __init__(self,
                 p=0.5,
                 seed=None,
                 name=None,
                 **kwargs):

        super(HorizontalFlip, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

    def apply(self, inputs, training=True):
        if training:
            return tf.image.flip_left_right(inputs)
        else:
            return inputs
