from dataclasses import dataclass
from typing import Any, Tuple, Union
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial


class Dataset(object):

    def __init__(self,
                 dataset_name: str,
                 train_split: Union[Tuple[str, float], str, tfds.Split],
                 eval_split: Union[Tuple[str, float], str, tfds.Split],
                 dataset_dir: str = None):

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.train_split = train_split
        self.eval_split = eval_split

        # Dataset attributes
        ds_info = tfds.builder(self.dataset_name).info
        if dataset_name == 'imagenet2012':
            self.ds_shape = [224, 224, 3]
        else:
            self.ds_shape = list(ds_info.features['image'].shape)
            has_none_val = True if None in self.ds_shape else False  
            assert not has_none_val, 'There should be no None in self.ds_shape'
        self.num_classes = ds_info.features['label'].num_classes
        self.num_train_examples = ds_info.splits[train_split].num_examples
        self.num_eval_examples = ds_info.splits[eval_split].num_examples
    
    def load_pretrain_datasets(self,
                               batch_size: int,
                               eval_batch_size: int,
                               train_aug: Any,
                               eval_aug: Any,
                               train_aug_2: Any = None,
                               drop_remainder=True,
                               seed: int = None,
                               dtype_policy: str = 'float32'):

        train_ds = tfds.load(self.dataset_name,
                             split=self.train_split,
                             data_dir=self.dataset_dir,
                             shuffle_files=True,
                             as_supervised=True)

        eval_ds = tfds.load(self.dataset_name,
                            split=self.eval_split,
                            data_dir=self.dataset_dir,
                            shuffle_files=False,
                            as_supervised=True)
        
        def preprocess_pretrain(image, label, aug_fn_1, aug_fn_2=None):
            image_size = self.ds_shape[0]
            xs = []
            for i in range(2):
                aug_fn = aug_fn_1
                if aug_fn_2 is not None:
                    if i == 1:
                        aug_fn = aug_fn_2
                aug_img = aug_fn(
                    image, training=True)
                aug_img = tf.reshape(
                    aug_img, [image_size, image_size, 3])
                xs.append(aug_img)
            image = tf.concat(xs, -1)
            return image, label
        
        preprocess_train = partial(
            preprocess_pretrain, aug_fn_1=train_aug, aug_fn_2=train_aug_2)
        preprocess_eval = partial(preprocess_pretrain, aug_fn_1=eval_aug)

        train_ds = self.batch_and_optimize(
            train_ds, batch_size, preprocess_train, drop_remainder)
        eval_ds = self.batch_and_optimize(
            eval_ds, eval_batch_size, preprocess_eval,
            drop_remainder, shuffle=False)
        return train_ds, eval_ds
    
    def load_finetune_datasets(self,
                               batch_size: int,
                               eval_batch_size: int,
                               train_aug: Any,
                               eval_aug: Any,
                               finetune_train_split=None,
                               finetune_eval_split=None,
                               drop_remainder=True,
                               seed: int = None,
                               dtype_policy: str = 'float32'):

        if finetune_train_split is None:
            finetune_train_split = self.train_split
        if finetune_eval_split is None:
            finetune_eval_split = self.eval_split

        def preprocess_finetune(image, label, aug_fn):
            dtype = DTYPE[dtype_policy]
            aug_img = aug_fn(
                image, training=True)
            aug_img = tf.reshape(
                aug_img, [image_size, image_size, 3])
            label = tf.cast(label, dtype)
            return aug_img, label
        
        preprocess_train = partial(preprocess_finetune, train_aug)
        preprocess_eval = partial(preprocess_finetune, eval_aug)

        train_ds = self.batch_and_optimize(
            train_ds, batch_size, preprocess_train, drop_remainder)
        eval_ds = self.batch_and_optimize(
            eval_ds, eval_batch_size, preprocess_eval, 
            drop_remainder, shuffle=False)
        return train_ds, eval_ds