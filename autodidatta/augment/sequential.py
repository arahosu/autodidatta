import tensorflow as tf
from tensorflow.keras import Sequential
import autodidatta.augment as A


class Augment(Sequential):

    def __init__(self,
                 layers=None,
                 name=None,
                 **kwargs):
        super(Augment, self).__init__(
            layers=layers,
            name=name,
            **kwargs
            )

    def call(self, image, seg=None, training=False):

        # Handle corner cases where self.layers is empty
        aug_image, aug_seg = image, seg

        for layer in self.layers:
            aug_image, aug_seg = layer.apply(
                image=image, seg=seg, training=training)
            image, seg = aug_image, aug_seg
        if seg is not None:
            return (aug_image, aug_seg)
        else:
            return aug_image


class SSLAugment(Augment):

    def __init__(self,
                 image_size,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 color_jitter_prob=0.8,
                 grayscale_prob=0.2,
                 horizontal_flip_prob=0.5,
                 gaussian_prob=0.0,
                 solarization_prob=0.0,
                 min_scale=0.08,
                 max_scale=1.0):
        
        super(SSLAugment, self).__init__()

        self.add(A.layers.RandomResizedCrop(
            image_size, image_size, scale=(min_scale, max_scale)))
        self.add(A.layers.HorizontalFlip(
            p=horizontal_flip_prob))
        self.add(A.layers.ColorJitter(
            brightness, contrast, saturation, hue, p=color_jitter_prob))
        self.add(A.layers.ToGray(p=grayscale_prob))
        self.add(A.layers.GaussianBlur(
            kernel_size=image_size // 10,
            sigma=tf.random.uniform([], 0.1, 2.0, dtype=tf.float32),
            padding='SAME',
            p=gaussian_prob))
        self.add(A.layers.Solarize(p=solarization_prob))
        


        