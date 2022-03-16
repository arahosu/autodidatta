import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import LAMB, AdamW
from math import sqrt

OPTIMIZER = {
    'lamb': LAMB,
    'adam': Adam,
    'adamw': AdamW,
    'sgd': SGD,
}

class WarmUpAndCosineDecay(LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule.

    Taken from Google Research SimCLR repository:
    https://github.com/google-research/simclr
    """

    def __init__(self,
                 base_learning_rate,
                 num_examples,
                 batch_size,
                 num_train_epochs,
                 warmup_epochs,
                 learning_rate_scaling='linear',
                 name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate_scaling = learning_rate_scaling
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                round(
                    self.warmup_epochs * self.num_examples // self.batch_size))
            if self.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * self.batch_size / 256.
            elif self.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * sqrt(self.batch_size)
            elif self.learning_rate_scaling == None:
                scaled_lr = self.base_learning_rate
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    self.learning_rate_scaling))
            learning_rate = (
                tf.cast(step, tf.float32) / float(warmup_steps) * scaled_lr
                if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = self.num_examples * self.num_train_epochs \
                // self.batch_size + 1
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate

    def get_config(self):
        return {
            'base_learning_rate': self.base_learning_rate,
            'num_examples': self.num_examples,
            }

def load_optimizer(optimizer_name,
                   learning_rate,
                   optimizer_configs=None):
    optimizer = OPTIMIZER[optimizer_name](
        learning_rate=learning_rate, **optimizer_configs)
    return optimizer