from datetime import datetime
import ml_collections
import os

from autodidatta.models.byol import BYOLMAWeightUpdate

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def load_callbacks(model_name,
                   log_dir,
                   weights_dir,
                   online_ft=True,
                   max_steps=None,
                   callback_configs: dict = None):
    
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = []
    kwargs = dict(callback_configs)

    if model_name == 'byol':
        assert max_steps is not None, 'max_steps should not be None if \
        model_name is byol'
        movingavg_cb = BYOLMAWeightUpdate(max_steps, **callback_configs)
        cb.append(movingavg_cb)

    if weights_dir is not None:
        weights_dir = os.path.join(weights_dir, time)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        weights_filename = model_name + '.hdf5'
        weights = ModelCheckpoint(
            os.path.join(weights_dir, weights_filename),
            save_weights_only=True,
            monitor='val_acc' if online_ft else 'similarity_loss',
            mode='max' if online_ft else 'min',
            save_best_only=True)
        cb.append(weights)

    if log_dir is not None:
        log_dir = os.path.join(log_dir, time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a callback for saving the logs in TensorBoard
        tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

        cb.append(tb_cb)
    
    return cb

    
    