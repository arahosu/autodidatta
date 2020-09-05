import tensorflow as tf
import tensorflow.keras.layers as tfkl

# not acc needed outside test
import numpy as np


def _is_shape_2d(input_tensor):
    """ Expect 2D if the input tensor is of shape
    (batch_size, channels, width, height)"""
    if len(input_tensor.shape) > 4:
        return False
    elif len(input_tensor.shape) <= 4:
        return True
    else:
        raise ValueError

def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   channels_last=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if channels_last:
        bn_axis = -1
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if _is_shape_2d(input_tensor):
        Conv = tfkl.Conv2D
        unit_kernel_size = (1, 1)
    else:
        Conv = tfkl.Conv3D
        unit_kernel_size = (1, 1, 1)

    x = Conv(filters1, unit_kernel_size,
             kernel_initializer='he_normal',
             name=conv_name_base + '2a')(input_tensor)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tfkl.Activation('relu')(x)

    x = Conv(filters2, kernel_size,
             padding='same',
             kernel_initializer='he_normal',
             name=conv_name_base + '2b')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tfkl.Activation('relu')(x)

    x = Conv(filters3, unit_kernel_size,
             kernel_initializer='he_normal',
             name=conv_name_base + '2c')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = tfkl.add([x, input_tensor])
    x = tfkl.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides,
               channels_last=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if channels_last:
        bn_axis = -1
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if _is_shape_2d(input_tensor):
        Conv = tfkl.Conv2D
        unit_kernel_size = (1, 1)
    else:
        Conv = tfkl.Conv3D
        unit_kernel_size = (1, 1, 1)

    x = Conv(filters1, kernel_size=unit_kernel_size, strides=strides,  # ks was (1, 1)
             kernel_initializer='he_normal',
             name=conv_name_base + '2a')(input_tensor)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tfkl.Activation('relu')(x)

    x = Conv(filters2, kernel_size, padding='same',
             kernel_initializer='he_normal',
             name=conv_name_base + '2b')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tfkl.Activation('relu')(x)

    x = Conv(filters3, kernel_size=unit_kernel_size,
             kernel_initializer='he_normal',
             name=conv_name_base + '2c')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv(filters3, kernel_size=unit_kernel_size, strides=strides,
                    kernel_initializer='he_normal',
                    name=conv_name_base + '1')(input_tensor)
    shortcut = tfkl.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = tfkl.add([x, shortcut])
    x = tfkl.Activation('relu')(x)
    return x


def ResNet50(include_top,
             input_shape,
             pooling=None,
             channels_last=True,
             classes=10,
             use_2d=True,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_shape: shape tuple
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    height, width, num_channels = input_shape
    
    if height <= 32 or width <= 32:
        conv1_strides = (1, 1)
    else:
        conv1_strides = (2, 2)

    # Set dimensionality 2D/3D
    if use_2d:
        ZeroPadding = tfkl.ZeroPadding2D
        Conv = tfkl.Conv2D
        MaxPooling = tfkl.MaxPooling2D
        GlobalAveragePooling = tfkl.GlobalAveragePooling2D
    else:
        ZeroPadding = tfkl.ZeroPadding3D
        Conv = tfkl.Conv3D
        MaxPooling = tfkl.MaxPooling3D
        GlobalAveragePooling = tfkl.GlobalAveragePooling3D

    img_input = tfkl.Input(shape=input_shape)

    if channels_last:
        bn_axis = -1
    else:
        bn_axis = 1

    x = ZeroPadding(padding=3,
                    name='conv1_pad')(img_input)
    
    if height <= 32 or width <= 32:
        conv_1_strides = 1 
    else:
        conv_1_strides = 2
    x = Conv(64, 7,
             strides=conv_1_strides,
             padding='valid',
             kernel_initializer='he_normal',
             name='conv1')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = tfkl.Activation('relu')(x)
    x = ZeroPadding(padding=1, name='pool1_pad')(x)
    x = MaxPooling(3, strides=2)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=1)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=1)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = GlobalAveragePooling(name='avg_pool')(x)
        x = tfkl.Dense(classes, activation='softmax', name='fc_output')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling()(x)
        elif pooling == 'max':
            x = GlobalAveragePooling()(x)

    model = tf.keras.Model(img_input, x, name='resnet50')

    return model


# if __name__ == "__main__":

#     # generate fake data
#     x = np.random.randint(0, 8, size=(100, 5, 5, 5, 8))
#     # y = np.random.randint(0, 8, size=(10, 8, 32, 32, 2))

#     num_classes = 8
#     y = np.random.choice([i for i in range(1, num_classes)], size=(500,))  # 0, 1, 2, 3, 4, 5, 6, 7
#     y = tf.keras.utils.to_categorical(y)  #, num_classes)

#     # build model  and compile it
#     use_2d = True
#     if use_2d:
#         x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])

#     res_model = ResNet50(
#                     include_top=True,
#                     input_shape=x.shape[1:],
#                     pooling=None,
#                     channels_last=True,
#                     classes=num_classes,
#                     use_2d=use_2d)

#     res_model.compile(tf.keras.optimizers.Adam(), loss="categorical_crossentropy")

#     res_model.summary()

#     # train ResMNet 3D model
#     res_model.fit(x, y, batch_size=16, epochs=3)
