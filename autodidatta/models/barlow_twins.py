from absl import app
from absl import flags
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger

from autodidatta.datasets.cifar10 import load_input_fn
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.mlp import projection_head
from autodidatta.utils.accelerator import setup_accelerator

