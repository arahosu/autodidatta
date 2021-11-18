from absl import flags

# Logging and callback flags
flags.DEFINE_bool(
    'save_weights', True,
    'Whether to save weights. If True, weights are saved in logdir')
flags.DEFINE_bool(
    'save_history', True,
    'Whether to save the training history.')
flags.DEFINE_string(
    'histdir', './training_logs',
    'Directory for where the training history is being saved')
flags.DEFINE_string(
    'logdir', './weights',
    'Directory for where the weights are being saved')
flags.DEFINE_string(
    'weights', None,
    'Directory for the trained model weights. Only used for finetuning')

FLAGS = flags.FLAGS  