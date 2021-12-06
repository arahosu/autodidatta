import tensorflow as tf
from math import sqrt


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule.
    Taken from Google Research SimCLR repository:
    https://github.com/google-research/simclr/blob/dec99a81a4ceccb0a5a893afecbc2ee18f1d76c3/tf2/model.py#L78
    """

    def __init__(self,
                 base_learning_rate,
                 num_examples,
                 batch_size,
                 warmup_epochs,
                 num_train_epochs,
                 learning_rate_scaling='linear',
                 name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.num_train_epochs = num_train_epochs
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
                step / float(warmup_steps) * scaled_lr
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
