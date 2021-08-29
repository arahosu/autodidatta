import tensorflow as tf
from tensorflow.config.experimental import list_logical_devices, \
    set_visible_devices, list_physical_devices


class LocalTPUClusterResolver(
    tf.distribute.cluster_resolver.TPUClusterResolver):
    """LocalTPUClusterResolver."""
    def __init__(self):
        self._tpu = ''
        self.task_type = 'worker'
        self.task_id = 0

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        return None

    def cluster_spec(self):
        return tf.train.ClusterSpec({})

    def get_tpu_system_metadata(self):
        return tf.tpu.experimental.TPUSystemMetadata(
            num_cores=8,
            num_hosts=1,
            num_of_cores_per_host=8,
            topology=None,
            devices=tf.config.list_logical_devices())

    def num_accelerators(self,
                         task_type=None,
                         task_id=None,
                         config_proto=None):
        return {'TPU': 8}


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
        print('Use TPU at {}'.format(device_name))
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=device_name)
        # resolver = LocalTPUClusterResolver()
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


def min_max_standardize(image):

    denom = tf.math.reduce_max(image) - tf.math.reduce_min(image)
    num = image - tf.math.reduce_min(image)

    return num / denom
