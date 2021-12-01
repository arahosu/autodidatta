from absl import flags

# Logging and callback flags
flags.DEFINE_string(
    'histdir', None,
    'Directory for where the training history is being saved')
flags.DEFINE_string(
    'logdir', None,
    'Directory for where the weights are being saved')

FLAGS = flags.FLAGS  