from absl import flags

# Dataset flags
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'stl10', 'imagenet'],
    'cifar10 (default), stl10, imagenet')
flags.DEFINE_string(
    'dataset_dir', None,
    'directory where the dataset is stored')

# Augmentation flags
flags.DEFINE_float(
    'brightness', 0.4,
    'random brightness factor')
flags.DEFINE_float(
    'contrast', 0.4,
    'random contrast factor')
flags.DEFINE_float(
    'saturation', 0.4,
    'random saturation factor')
flags.DEFINE_float(
    'hue', 0.1,
    'random hue factor')
flags.DEFINE_list(
    'gaussian_prob', [0.0, 0.0],
    'probability of applying gaussian blur augmentation')
flags.DEFINE_list(
    'solarization_prob', [0.0, 0.0],
    'probability of applying solarization augmentation')
flags.DEFINE_bool(
    'use_simclr_augment', False,
    'Use the data augmentation ')

FLAGS = flags.FLAGS  

def parse_augmentation_flags():
    kwargs = {
        'brightness': FLAGS.brightness,
        'contrast': FLAGS.contrast,
        'saturation': FLAGS.saturation,
        'hue': FLAGS.hue
    }
    return kwargs
