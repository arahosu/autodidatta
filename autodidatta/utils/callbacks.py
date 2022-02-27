from datetime import datetime
import os

from autodidatta.models.byol import BYOLMAWeightUpdate

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


def load_callbacks(model_name,
                   history_dir,
                   weights_dir,
                   weights_filename,
                   history_filename,
                   online_ft=True,
                   callback_configs=None):
    
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    cb = []

    if model_name == 'byol':
        movingavg_cb = BYOLMAWeightUpdate(**callback_configs)
        cb.append(movingavg_cb)

    if weights_dir is not None:
        weights_dir = os.path.join(weights_dir, time)
        os.mkdir(weights_dir)
        weights = ModelCheckpoint(
            os.path.join(weights_dir, weights_filename),
            save_weights_only=True,
            monitor='val_acc' if online_ft else 'similarity_loss',
            mode='max' if online_ft else 'min',
            save_best_only=True)
        cb.append(weights)

    if history_dir is not None:
        history_dir = os.path.join(history_dir, time)
        os.mkdir(history_dir)

        # Create a callback for saving the training results into a csv file
        csv_logger = CSVLogger(
            os.path.join(history_dir, history_filename))

        cb.append(csv_logger)
    
    return cb

    
    