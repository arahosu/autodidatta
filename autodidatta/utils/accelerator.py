import tensorflow as tf
from tensorflow.config import list_logical_devices, \
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
        print('Use TPU at {}'.format(device_name))
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=device_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

    return strategy
