import tensorfow as tf
import tensorflow.keras.layers as tfkl

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

    x = tfkl.Conv2D(filters1, (1, 1),
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2a')(input_tensor)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tfkl.Activation('relu')(x)

    x = tfkl.Conv2D(filters2, kernel_size,
                    padding='same',
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2b')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tfkl.Activation('relu')(x)

    x = tfkl.Conv2D(filters3, (1, 1),
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
               strides=(2, 2),
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

    x = tfkl.Conv2D(filters1, (1, 1), strides=strides,
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2a')(input_tensor)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = tfkl.Activation('relu')(x)

    x = tfkl.Conv2D(filters2, kernel_size, padding='same',
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2b')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = tfkl.Activation('relu')(x)

    x = tfkl.Conv2D(filters3, (1, 1),
                    kernel_initializer='he_normal',
                    name=conv_name_base + '2c')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = tfkl.Conv2D(filters3, (1, 1), strides=strides,
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

    img_input = tfkl.Input(shape=input_shape)

    if channels_last:
        bn_axis = -1
    else:
        bn_axis = 1

    x = tfkl.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = tfkl.Conv2D(64, (7, 7),
                    strides=conv1_strides,
                    padding='valid',
                    kernel_initializer='he_normal',
                    name='conv1')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = tfkl.Activation('relu')(x)
    x = tfkl.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = tfkl.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = tfkl.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tfkl.Dense(classes, activation='softmax', name='fc_output')(x)
    else:
        if pooling == 'avg':
            x = tfkl.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tfkl.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(img_input, x, name='resnet50')

    return model
