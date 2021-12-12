from absl import app
from absl import flags
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow_addons.optimizers import AdamW
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger

import autodidatta.augment as A
from autodidatta.datasets import Dataset
from autodidatta.flags import dataset_flags, training_flags, utils_flags
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.utils.accelerator import setup_accelerator

flags.DEFINE_integer(
    'percentage_data',
    100,
    'percentage of training data to be used during downstream evaluation')
flags.DEFINE_string(
    'weights', None,
    'Directory for the trained model weights. Only used for finetuning')

flags.FLAGS.set_default('train_epochs', 300)
flags.FLAGS.set_default('batch_size', 256)
FLAGS = flags.FLAGS

def main(argv):

    del argv

    # Choose accelerator 
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)
    
    # Choose whether to train with float32 or bfloat16 precision
    if FLAGS.use_bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Select dataset
    train_split = 'train[:%s%%]' %(FLAGS.percentage_data)

    if FLAGS.dataset in ['cifar10', 'cifar100']:
        image_size = 32
        validation_split = 'test'
        num_classes = 10 if FLAGS.dataset == 'cifar10' else 100
    elif FLAGS.dataset == 'stl10':
        image_size = 96
        validation_split = 'test'
        num_classes = 10
    elif FLAGS.dataset == 'imagenet2012':
        assert FLAGS.dataset_dir is not None, 'for imagenet2012, \
            dataset direcotry must be specified'
        image_size = 224
        validation_split = 'validation'
        num_classes = 1000
    else:
        raise NotImplementedError("other datasets have not yet been implmented")

    # Define augmentation functions
    augment_kwargs = dataset_flags.parse_augmentation_flags()
    if FLAGS.use_simclr_augment:
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
    else:
        if FLAGS.dataset in ['cifar10', 'cifar100']:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.247, 0.243, 0.261]
        elif FLAGS.dataset in ['stl10', 'imagenet2012']:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

    aug_fn = tf.keras.Sequential(
        [A.layers.RandomResizedCrop(
            image_size, image_size),
         A.layers.HorizontalFlip(p=0.5),
         A.layers.GaussianBlur(
            kernel_size=image_size // 10,
            sigma=tf.random.uniform([], 0.1, 2.0, dtype=tf.float32),
            padding='SAME',
            p=FLAGS.gaussian_prob[0]),
        A.layers.Normalize(mean=mean, std=std)])

    # Define dataloaders
    train_loader = Dataset(
        FLAGS.dataset,
        train_split,
        FLAGS.dataset_dir,
        aug_fn, None)
    validation_loader = Dataset(
        FLAGS.dataset,
        validation_split,
        FLAGS.dataset_dir,
        aug_fn, None)

    # Define datasets from the dataloaders
    train_ds = train_loader.load(
        FLAGS.batch_size,
        image_size,
        True,
        False,
        use_bfloat16=FLAGS.use_bfloat16)

    validation_ds = validation_loader.load(
        FLAGS.batch_size,
        image_size,
        False,
        False,
        use_bfloat16=FLAGS.use_bfloat16)
    
    # Get number of examples from dataloaders
    num_train_examples = train_loader.dataset_size
    num_val_examples = validation_loader.dataset_size
    steps_per_epoch = num_train_examples // FLAGS.batch_size
    validation_steps = num_val_examples // FLAGS.batch_size
    ds_shape = (image_size, image_size, 3)

    with strategy.scope():
        # Define backbone
        if FLAGS.backbone == 'resnet50':
            backbone = ResNet50(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet34':
            backbone = ResNet34(input_shape=ds_shape)
        elif FLAGS.backbone == 'resnet18':
            backbone = ResNet18(input_shape=ds_shape)
        else:
            raise NotImplementedError("other backbones have not yet been implemented")

        # Load weights and freeze the backbone
        if FLAGS.weights is not None:
            backbone.load_weights(FLAGS.weights)
            backbone.trainable = False
        else:
            print("Training {} in a fully supervised setting".format(FLAGS.backbone))

        # load classifier for downstream task evaluation
        classifier = training_flags.load_classifier(num_classes)
        loss = tf.keras.losses.sparse_categorical_crossentropy
        metrics = ['acc']

        model = tf.keras.Sequential(
            [backbone,
             classifier])

        optimizer, _ = training_flags.load_optimizer(num_train_examples)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)
        model.build((None, *ds_shape))

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = None

    if FLAGS.histdir is not None:
        histdir = os.path.join(FLAGS.histdir, time)
        os.mkdir(histdir)

        # Create a callback for saving the training results into a csv file
        histfile = 'finetune_results.csv'
        csv_logger = CSVLogger(os.path.join(histdir, histfile))
        cb = [csv_logger]

        # Save flag params in a flag file in the same subdirectory
        flagfile = os.path.join(histdir, 'train_flags.cfg')
        FLAGS.append_flags_into_file(flagfile)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.train_epochs,
        validation_data=validation_ds,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=cb)

if __name__ == '__main__':
    app.run(main)