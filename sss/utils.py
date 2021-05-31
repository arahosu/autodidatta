import tensorflow as tf
from tensorflow.config.experimental import list_logical_devices, \
    set_visible_devices, list_physical_devices


def setup_accelerator(use_gpu, num_cores, device_name=None):

    """A helper function for setting up single/multi-GPU or TPU training
    """

    if use_gpu:
        print('Using GPU...')
        # strategy requires: export TF_FORCE_GPU_ALLOW_GROWTH=true
        # to be written in cmd
        if num_cores == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.MirroredStrategy()
        gpus = list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    set_visible_devices(gpu, 'GPU')
                    logical_gpus = list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,",
                          len(logical_gpus), "Logical GPU")
                except RuntimeError as e:
                    # Visible devices must be set before GPUs are initialized
                    print(e)
    else:
        print('Use TPU at %s',
              device_name if device_name is not None else 'local')
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=device_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

    return strategy


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 steps_per_epoch,
                 initial_learning_rate,
                 min_learning_rate,
                 drop,
                 epochs_drop,
                 warmup_epochs):
        super(LearningRateSchedule, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.drop = drop
        self.epochs_drop = epochs_drop
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        lrate = self.initial_learning_rate
        if self.warmup_epochs >= 1:
            lrate *= lr_epoch / self.warmup_epochs
        epochs_drop = [self.warmup_epochs] + self.epochs_drop
        for index, start_epoch in enumerate(epochs_drop):
            lrate = tf.where(
                lr_epoch >= start_epoch,
                self.update_lr(index, self.min_learning_rate),
                lrate)
        return lrate

    def update_lr(self, idx, min_lr):

        new_lr = self.initial_learning_rate * self.drop**idx
        if new_lr < min_lr:
            new_lr = min_lr

        return new_lr

    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'initial_learning_rate': self.initial_learning_rate,
        }
