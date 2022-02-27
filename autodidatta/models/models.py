from autodidatta.models.simclr import SimCLR
from autodidatta.models.simsiam import SimSiam
from autodidatta.models.byol import BYOL
from autodidatta.models.barlow_twins import BarlowTwins

from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50

import tensorflow as tf
import tensorflow.keras.layers as tfkl


MODEL_CLS = {
    'simclr': SimCLR,
    'simsiam': SimSiam,
    'byol': BYOL,
    'barlow_twins': BarlowTwins
}

BACKBONE = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50
}

def get_model_cls(input_shape,
                  model_name,
                  model_configs,
                  classifier=None):
    
    kwargs = dict(model_configs)
    backbone_name = kwargs.pop('backbone', None)
    backbone = BACKBONE[backbone_name](input_shape)
    model_cls = MODEL_CLS[model_name]

    return model_cls(backbone, classifier=classifier, **kwargs)


def load_classifier(num_classes,
                    linear_eval=True,
                    strategy=None):

    if linear_eval:
        classifier = tf.keras.Sequential(
            [tfkl.Flatten(),
             tfkl.Dense(num_classes, activation='softmax')],
             name='classifier')
    else:
        if strategy is None or strategy.num_replicas_in_sync == 1:
            BatchNorm = tfkl.BatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5)
        else:
            BatchNorm = tfkl.experimental.SyncBatchNormalization(
                axis=-1, momentum=0.9, epsilon=1.001e-5)

        classifier = tf.keras.Sequential(
            [tfkl.Flatten(),
             tfkl.Dense(512, use_bias=False),
             BatchNorm,
             tfkl.ReLU(),
             tfkl.Dense(num_classes, activation='softmax')],
             name='classifier')
    return classifier