from autodidatta.augment.layers.base import DualOps
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.image import sample_distorted_bounding_box


class RandomZoom(DualOps):
    # TODO: Implement RandomZoom without importing RandomZoom
    def __init__(self,
                 height_factor,
                 width_factor,
                 fill_mode='reflect',
                 interpolation='bilinear',
                 p=1.0,
                 seed=None,
                 name=None,
                 fill_value=0.0,
                 **kwargs):
        super(RandomZoom, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        self.op = tfkl.RandomZoom(
            height_factor, width_factor, fill_mode,
            interpolation, seed, fill_value=fill_value
        )

    def call(self, inputs, training=True):
        return self.op(inputs, training=training)


class RandomRotate(DualOps):
    # TODO: Implement RandomZoom without importing RandomRotate
    def __init__(self,
                 factor,
                 fill_mode='reflect',
                 interpolation='bilinear',
                 p=0.5,
                 seed=None,
                 name=None,
                 fill_value=0.0,
                 **kwargs):
        super(RandomRotate, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

        self.op = tfkl.RandomRotation(
            factor, fill_mode, interpolation,
            seed, fill_value=fill_value
        )

    def call(self, inputs, training=True):
        return self.op(inputs, training=training)


class RandomResizedCrop(DualOps):

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

    def call(self, inputs, training=True):
        image_dtype = inputs.dtype
        if training:
            bbox = tf.constant(
                [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
            aspect_ratio = self.height / self.width
            ratio = tuple([aspect_ratio*x for x in self.ratio])

            distorted_bb = sample_distorted_bounding_box(
                image_size=inputs.shape,
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
        else:
            return inputs


class HorizontalFlip(DualOps):

    def __init__(self,
                 p=0.5,
                 seed=None,
                 name=None,
                 **kwargs):

        super(HorizontalFlip, self).__init__(
            p=p, seed=seed, name=name, **kwargs
        )

    def call(self, inputs, training=True):
        if training:
            return tf.image.flip_left_right(inputs)
        else:
            return inputs
