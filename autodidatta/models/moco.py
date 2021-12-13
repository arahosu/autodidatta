from absl import app
from absl import flags
from datetime import datetime
import math
import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

import autodidatta.augment as A
from autodidatta.datasets import Dataset
from autodidatta.flags import dataset_flags, training_flags, utils_flags
from autodidatta.models.base import BaseModel
from autodidatta.models.networks.resnet import ResNet18, ResNet34, ResNet50
from autodidatta.models.networks.mlp import projection_head, predictor_head
from autodidatta.utils.loss import moco_loss
from autodidatta.utils.accelerator import setup_accelerator

# moco flags
flags.DEFINE_float(
    'momentum', 0.99, 'momentum parameter for target network update')
flags.DEFINE_integer(
    'num_negatives', 32768, 'number of negatives to be used')

# Redefine default value
flags.FLAGS.set_default(
    'proj_hidden_dim', 2048)
flags.FLAGS.set_default(
    'output_dim', 128)
flags.FLAGS.set_default(
    'use_bfloat16', False)

FLAGS = flags.FLAGS

class MomentumUpdate(tf.keras.callbacks.Callback):

    def __init__(self,
                 momentum,
                 num_negatives):
        super(MomentumUpdate, self).__init__()

        self.momentum = momentum
        self.num_negatives = num_negatives

    def on_train_batch_end(self, batch, logs=None):
        for online_layer, target_layer in zip(
            self.model.backbone.layers,
            self.model.target_backbone.layers):
            if hasattr(target_layer, 'kernel'):
                target_layer.kernel.assign(self.momentum * target_layer.kernel 
                                           + (1 - self.momentum) * online_layer.kernel)
            if hasattr(target_layer, 'bias'):
                target_layer.bias.assign(self.momentum * target_layer.bias 
                                         + (1 - self.momentum) * online_layer.bias)
            if hasattr(target_layer, 'gamma'):
                target_layer.gamma.assign(self.momentum * target_layer.gamma 
                                         + (1 - self.momentum) * online_layer.gamma)
            if hasattr(target_layer, 'beta'):
                target_layer.beta.assign(self.momentum * target_layer.beta 
                                         + (1 - self.momentum) * online_layer.beta)

        for online_layer, target_layer in zip(
            self.model.projector.layers,
            self.model.target_projector.layers):
            if hasattr(target_layer, 'kernel'):
                target_layer.kernel.assign(self.momentum * 
                target_layer.kernel + (1 - self.momentum) * 
                online_layer.kernel)
            if hasattr(target_layer, 'bias'):
                if target_layer.bias is not None:
                    target_layer.bias.assign(self.momentum * 
                    target_layer.bias + (1 - self.momentum) * 
                    online_layer.bias)
            if hasattr(target_layer, 'gamma'):
                target_layer.gamma.assign(self.momentum * target_layer.gamma 
                + (1 - self.momentum) * online_layer.gamma)
            if hasattr(target_layer, 'beta'):
                target_layer.beta.assign(self.momentum * target_layer.beta 
                + (1 - self.momentum) * online_layer.beta)

        key = logs.pop('key')
        self.model.queue = tf.concat([tf.transpose(key, perm=[0, 2, 1]), self.model.queue], axis=-1)
        self.model.queue = self.model.queue[:,:, :self.num_negatives]


class Moco(BaseModel):
    def __init__(self,
                 backbone,
                 projector,
                 queue_dim,
                 num_negatives,
                 loss_temperature,
                 classifier=None):
        
        super(Moco, self).__init__(
            backbone=backbone,
            projector=projector,
            predictor=None,
            classifier=classifier
        )

        self.target_backbone = tf.keras.models.clone_model(backbone)
        self.target_projector = tf.keras.models.clone_model(projector)

        _queue = tf.random.normal([2, queue_dim, num_negatives])
        _queue, _ = tf.linalg.normalize(_queue, axis=1)
        self.queue = self.add_weight(
            name='queue',
            shape=(2, queue_dim, num_negatives),
            initializer=tf.keras.initializers.Constant(_queue),
            trainable=False)
        self.loss_temperature = loss_temperature

    def build(self, input_shape):

        self.backbone.build(input_shape)
        self.projector.build(
            self.backbone.compute_output_shape(input_shape))
        
        self.target_backbone.build(input_shape)
        self.target_projector.build(
            self.target_backbone.compute_output_shape(input_shape))

        if self.classifier is not None:
            self.classifier.build(
                self.backbone.compute_output_shape(input_shape))

        self.built = True
    
    def shared_step(self, data, training, return_key=False):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data 
        num_channels = int(x.shape[-1] // 2)

        xi = x[..., :num_channels]
        xj = x[..., num_channels:]

        k1 = self.target_projector(
            self.target_backbone(xi,
            training=False),
            training=False)
        k1 = tf.math.l2_normalize(k1, axis=-1)

        k2 = self.target_projector(
            self.target_backbone(xj,
            training=False),
            training=False)
        k2 = tf.math.l2_normalize(k2, axis=-1)

        q1 = self.projector(
            self.backbone(xi,
            training=training),
            training=training)
        q1 = tf.math.l2_normalize(q1, axis=-1)

        q2 = self.projector(
            self.backbone(xj,
            training=training),
            training=training)
        q2 = tf.math.l2_normalize(q2, axis=-1)

        loss = self.loss_fn(q1, k2, self.queue[1], self.loss_temperature, self.distribute_strategy)
        loss += self.loss_fn(q2, k1, self.queue[0], self.loss_temperature, self.distribute_strategy)
        loss /= 2

        loss = loss + sum(self.losses)

        loss /= self.distribute_strategy.num_replicas_in_sync

        if return_key:
            key = tf.stack([self.update_queue(k1), self.update_queue(k2)])
            return loss, key
        else:
            return loss
    
    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss, key = self.shared_step(data, training=True, return_key=True)
        
        trainable_variables = self.backbone.trainable_variables + \
            self.projector.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))

        if self.classifier is not None:
            self.finetune_step(data)
            metrics_results = {m.name: m.result() for m in self.metrics}
            results = {'similarity_loss': loss, **metrics_results}
        else:
            results = {'similarity_loss': loss}
        
        if not 'key' in results:
            results.update({'key': key})

        return results
    
    def concat_fn(self, strategy, key_per_replica):
        return tf.concat(key_per_replica, axis=0)

    def unshuffle_bn(self, key, batch_size, unshuffle_idx):
        _replica_context = tf.distribute.get_replica_context()
        key_all_replica = _replica_context.merge_call(
            self.concat_fn, args=(key,))
        unshuffle_idx_all_replica = _replica_context.merge_call(
            self.concat_fn, args=(unshuffle_idx,))
        new_key_list = []
        for idx in unshuffle_idx_all_replica:
            new_key_list.append(tf.expand_dims(key_all_replica[idx], axis=0))
        key_orig = tf.concat(tuple(new_key_list), axis=0)
        key = key_orig[(batch_size//self.distribute_strategy.num_replicas_in_sync)*(_replica_context.replica_id_in_sync_group):
                        (batch_size//self.distribute_strategy.num_replicas_in_sync)*(_replica_context.replica_id_in_sync_group+1)]
        return key
    
    def reduce_key(self, key):
        _replica_context = tf.distribute.get_replica_context()
        all_key = _replica_context.merge_call(self.concat_fn, args=(key,))
        return all_key
    
    def update_queue(self, key):
        if self.distribute_strategy.num_replicas_in_sync > 1:
            key = self.reduce_key(key)
        return key


def main(argv):

    del argv

    # Choose accelerator 
    strategy = setup_accelerator(
        FLAGS.use_gpu, FLAGS.num_cores, FLAGS.tpu)
    
    # Choose whether to train with float32 or bfloat16 precision
    if FLAGS.use_bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    # Select dataset
    if FLAGS.dataset in ['cifar10', 'cifar100']:
        image_size = 32
        train_split = 'train'
        validation_split = 'test'
        num_classes = 10 if FLAGS.dataset == 'cifar10' else 100
    elif FLAGS.dataset == 'stl10':
        image_size = 96
        train_split = 'train' if not online_ft else 'unlabelled'
        validation_split = 'test'
        num_classes = 10
    elif FLAGS.dataset == 'imagenet2012':
        assert FLAGS.dataset_dir is not None, 'for imagenet2012, \
            dataset direcotry must be specified'
        image_size = 224
        train_split = 'train'
        validation_split = 'validation'
        num_classes = 1000
    else:
        raise NotImplementedError("other datasets have not yet been implmented")

    # Define augmentation functions
    augment_kwargs = dataset_flags.parse_augmentation_flags()
    if FLAGS.use_simclr_augment:
        aug_fn = A.SimCLRAugment
    else:
        aug_fn = A.SSLAugment

    aug_fn_1 = aug_fn(
        image_size=image_size,
        gaussian_prob=FLAGS.gaussian_prob[0],
        solarization_prob=FLAGS.solarization_prob[0],
        **augment_kwargs)
    aug_fn_2 = aug_fn(
        image_size=image_size,
        gaussian_prob=FLAGS.gaussian_prob[1],
        solarization_prob=FLAGS.solarization_prob[1],
        **augment_kwargs)

    # Define dataloaders
    train_loader = Dataset(
        FLAGS.dataset,
        train_split,
        FLAGS.dataset_dir,
        aug_fn_1, aug_fn_2)
    validation_loader = Dataset(
        FLAGS.dataset,
        validation_split,
        FLAGS.dataset_dir,
        aug_fn_1, aug_fn_2)

    # Define datasets from the dataloaders
    train_ds = train_loader.load(
        FLAGS.batch_size,
        image_size,
        True,
        True,
        use_bfloat16=FLAGS.use_bfloat16)

    validation_ds = validation_loader.load(
        FLAGS.batch_size,
        image_size,
        False,
        True,
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

        # If online finetuning is enabled
        if FLAGS.online_ft:
            assert FLAGS.dataset != 'stl10', \
                'Online finetuning is not supported for stl10'

            # load classifier for downstream task evaluation
            classifier = training_flags.load_classifier(num_classes)

            finetune_loss = tf.keras.losses.sparse_categorical_crossentropy
            metrics = ['acc']
        else:
            classifier = None

        model = Moco(backbone=backbone,
                     projector=projection_head(
                         hidden_dim=FLAGS.proj_hidden_dim,
                         output_dim=FLAGS.output_dim,
                         num_layers=FLAGS.num_head_layers,
                         batch_norm_output=False),
                     queue_dim=FLAGS.output_dim,
                     num_negatives=FLAGS.num_negatives,
                     loss_temperature=FLAGS.loss_temperature,
                     classifier=classifier)

        # load_optimizer
        optimizer, ft_optimizer = training_flags.load_optimizer(num_train_examples)

        if FLAGS.online_ft:
            model.compile(
                optimizer=optimizer,
                loss_fn=moco_loss,
                ft_optimizer=ft_optimizer,
                loss=finetune_loss,
                metrics=metrics)
        else:
            model.compile(
                optimizer=optimizer,
                loss_fn=moco_loss)

        # Build the model
        model.build((None, *ds_shape))

    # Define checkpoints
    time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Moving Average Weight Update Callback
    movingavg_cb = MomentumUpdate(
        FLAGS.momentum,
        FLAGS.num_negatives
    )
    cb = [movingavg_cb]

    if FLAGS.logdir is not None:
        logdir = os.path.join(FLAGS.logdir, time)
        os.mkdir(logdir)
        weights_file = 'moco_weights.hdf5'
        weights = ModelCheckpoint(
            os.path.join(logdir, weights_file),
            save_weights_only=True,
            monitor='val_acc' if FLAGS.online_ft else 'similarity_loss',
            mode='max' if FLAGS.online_ft else 'min',
            save_best_only=True)
        cb.append(weights)
    if FLAGS.histdir is not None:
        histdir = os.path.join(FLAGS.histdir, time)
        os.mkdir(histdir)

        # Create a callback for saving the training results into a csv file
        histfile = 'moco_results.csv'
        csv_logger = CSVLogger(os.path.join(histdir, histfile))

        cb.append(csv_logger)

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
