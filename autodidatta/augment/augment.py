import ml_collections
import copy
from autodidatta.augment.sequential import SSLAugment, SimCLRAugment

import tensorflow as tf
import tensorflow.keras.layers as tfkl

def load_aug_fn_pretrain(dataset_name: str,
                         model_name: str,
                         image_size: int,
                         aug_configs: dict,
                         seed: int = None):

    if model_name == 'simclr':
        aug_fn = SimCLRAugment
    else:
        aug_fn = SSLAugment

    kwargs = dict(aug_configs)
    gaussian_prob = kwargs.pop('gaussian_prob', None)
    solarization_prob = kwargs.pop('solarization_prob', None)

    aug_fn_1 = aug_fn(
        image_size=image_size,
        gaussian_prob=gaussian_prob[0],
        solarization_prob=solarization_prob[0],
        seed=seed,
        **kwargs)
    aug_fn_2 = aug_fn(
        image_size=image_size,
        gaussian_prob=gaussian_prob[1],
        solarization_prob=solarization_prob[1],
        seed=seed,
        **kwargs)

    eval_aug_fn = tf.keras.Sequential()
    eval_aug_fn.add(
        tfkl.Normalization(
            mean=kwargs['mean'], variance=tf.math.square(kwargs['std'])))
    if dataset_name == 'imagenet2012':
        eval_aug_fn.add(tfkl.CenterCrop(image_size, image_size))

    return aug_fn_1, aug_fn_2, eval_aug_fn