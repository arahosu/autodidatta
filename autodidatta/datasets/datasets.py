from dataclasses import dataclass
from typing import Any, Tuple, Union
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class Dataset:
    dataset_name: str
    split: Union[Tuple[str, float], str, tfds.Split]
    dataset_dir: str = None
    augmentation_fn: Any = None
    augmentation_fn_2: Any = None

    def load(self,
             batch_size,
             image_size,
             shuffle,
             pre_train,
             drop_remainder=True,
             use_bfloat16=False):
        
        ds_info = tfds.builder(self.dataset_name).info
        dataset_size = ds_info.splits[self.split].num_examples
        self.dataset_size = dataset_size

        is_training=True if ('train' in self.split or 'unlabelled' in self.split) else False

        def preprocess(image, label):
            dtype = tf.bfloat16 if use_bfloat16 else tf.float32
            if not pre_train:
                label = tf.cast(label, dtype)

            if pre_train:
                xs = []
                for i in range(2):
                    augmentation_fn = self.augmentation_fn
                    assert augmentation_fn is not None, 'If set to pre_training mode, \
                        augmentation_fn should not be None'
                    if self.augmentation_fn_2 is not None:
                        if i == 1:
                            augmentation_fn = self.augmentation_fn_2
                    aug_img = augmentation_fn(image, training=is_training)
                    aug_img = tf.reshape(aug_img, [image_size, image_size, 3])
                    xs.append(aug_img)
                image = tf.concat(xs, -1)
                return image, label

            else:
                image = self.augmentation_fn(image, training=is_training)
                image = tf.reshape(image, [image_size, image_size, 3])
                return image, label

        dataset = tfds.load(self.dataset_name,
                            split=self.split,
                            data_dir=self.dataset_dir,
                            shuffle_files=True if shuffle else False,
                            as_supervised=True)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
        