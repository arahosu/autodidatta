from absl import flags

# Dataset flags
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['cifar10', 'cifar100', 'stl10', 'imagenet2012'],
    'cifar10 (default), cifar100, stl10, imagenet2012')
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
    'use_simclr_augment', True,
    'Use the data augmentation pipeline defined in SimCLR')

FLAGS = flags.FLAGS  

def parse_augmentation_flags():
    if not FLAGS.use_simclr_augment:
        if FLAGS.dataset in ['stl10', 'imagenet2012']:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        elif FLAGS.dataset in ['cifar10', 'cifar100']:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.247, 0.243, 0.261]
    else:
        mean = [0., 0., 0.]
        std = [1., 1., 1.]

    kwargs = {
        'brightness': FLAGS.brightness,
        'contrast': FLAGS.contrast,
        'saturation': FLAGS.saturation,
        'hue': FLAGS.hue,
        'mean': mean,
        'std': std
    }
    return kwargs
