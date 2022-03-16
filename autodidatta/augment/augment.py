import ml_collections
import copy
from autodidatta.augment.sequential import SSLAugment
import autodidatta.augment as A

import tensorflow as tf
import tensorflow.keras.layers as tfkl


def load_aug_fn_pretrain(dataset_name: str,
                         image_size: int,
                         aug_configs: dict,
                         seed: int = None):

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
        A.layers.Normalize(
            mean=kwargs['mean'], std=kwargs['std']))
    if dataset_name == 'imagenet2012':
        eval_aug_fn.add(A.layers.CentralCrop(image_size, image_size, 0.875))

    return aug_fn_1, aug_fn_2, eval_aug_fn