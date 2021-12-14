from autodidatta.augment.layers.base import BaseOps
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.image import sample_distorted_bounding_box
import matplotlib.pyplot as plt
import math


class RandomResizedCrop(BaseOps):

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

    def apply(self, inputs, training=True):
        image_dtype = inputs.dtype
        if training:
            return self._op(inputs)
        else:
            return inputs


class HorizontalFlip(BaseOps):

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


class Patches(tfkl.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = layers.Reshape((-1, patch_size * patch_size * 3))

    def call(self, images):
        # Create patches from the input images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.utils.array_to_img(images[idx]))
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(keras.utils.img_to_array(patch_img))
            plt.axis("off")
        plt.show()

        return idx

    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        n = int(math.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches, self.patch_size, self.patch_size, 3))
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed

    
    