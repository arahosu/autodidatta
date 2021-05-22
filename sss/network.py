import tensorflow as tf
import tensorflow.keras.layers as tfkl


def projection_head(hidden_dim=512,
                    output_dim=128,
                    num_layers=1,
                    batch_norm_output=False
                    ):

    model = tf.keras.Sequential()
    model.add(tfkl.GlobalAveragePooling2D())
    model.add(tfkl.Flatten())

    for _ in range(num_layers):
        model.add(tfkl.Dense(hidden_dim, use_bias=False))
        model.add(tfkl.BatchNormalization())
        model.add(tfkl.ReLU())

    model.add(tfkl.Dense(output_dim, use_bias=True))
    if batch_norm_output:
        model.add(tfkl.BatchNormalization())

    return model


def predictor_head(hidden_dim=512,
                   output_dim=128,
                   num_layers=1
                   ):

    model = tf.keras.Sequential()
    model.add(tfkl.Flatten())

    for _ in range(num_layers):
        model.add(tfkl.Dense(hidden_dim, use_bias=False))
        model.add(tfkl.BatchNormalization())
        model.add(tfkl.ReLU())

    model.add(tfkl.Dense(output_dim, use_bias=True))

    return model
