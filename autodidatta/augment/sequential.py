import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import Sequential
import autodidatta.augment as A


class SSLAugment(Sequential):

    def __init__(self,
                 image_size,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.247, 0.243, 0.261],
                 color_jitter_prob=0.8,
                 grayscale_prob=0.2,
                 horizontal_flip_prob=0.5,
                 gaussian_prob=0.0,
                 solarization_prob=0.0,
                 min_scale=0.08,
                 max_scale=1.0,
                 seed=None):
        
        super(SSLAugment, self).__init__()
    
        self.seed = seed

        self.add(A.layers.RandomResizedCrop(
            image_size, image_size, scale=(min_scale, max_scale)))
        self.add(A.layers.ColorJitter(
            brightness, contrast, saturation, hue, p=color_jitter_prob))
        self.add(A.layers.ToGray(p=grayscale_prob))
        self.add(A.layers.GaussianBlur(
            kernel_size=image_size // 10,
            sigma=tf.random.uniform(
                [], 0.1, 2.0, dtype=tf.float32),
            padding='SAME',
            p=gaussian_prob))
        self.add(A.layers.Solarize(p=solarization_prob))
        self.add(A.layers.HorizontalFlip(
            p=horizontal_flip_prob))
        self.add(A.layers.Normalize(mean=mean, std=std))


class SimCLRAugment(Sequential):

    def __init__(self,
                 image_size,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 color_jitter_prob=0.8,
                 grayscale_prob=0.2,
                 horizontal_flip_prob=0.5,
                 gaussian_prob=0.0,
                 solarization_prob=0.0,
                 min_scale=0.08,
                 max_scale=1.0,
                 seed=None):
        
        super(SimCLRAugment, self).__init__()

        self.seed = seed

        self.add(A.layers.Normalize(mean=mean, std=std))
        self.add(A.layers.RandomResizedCrop(
            image_size, image_size, scale=(min_scale, max_scale)))
        self.add(A.layers.ColorJitter(
            brightness, contrast, saturation, hue,
            p=color_jitter_prob, clip_value=True))
        self.add(A.layers.ToGray(p=grayscale_prob))
        self.add(A.layers.GaussianBlur(
            kernel_size=image_size // 10,
            sigma=tf.random.uniform(
                [], 0.1, 2.0, dtype=tf.float32),
            padding='SAME',
            p=gaussian_prob))
        self.add(A.layers.Solarize(
            threshold=0.5, p=solarization_prob))
        self.add(A.layers.HorizontalFlip(
            p=horizontal_flip_prob))