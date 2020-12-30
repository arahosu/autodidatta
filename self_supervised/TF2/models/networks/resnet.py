import tensorflow as tf
import tensorflow.keras.layers as tfkl

def basic_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """Basic conv block for ResNet18 & ResNet34.
    Arguments:
    x: input tensor.
    filters: integer, filters of the conv layers.
    stride: default 1, stride of the first layer.
    kernel_size: default 3, kernel size of the conv layers.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
    Returns:
    Output tensor for the basic conv block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = tfkl.Conv2D(filters, 1,
                               kernel_initializer='he_normal',
                               strides=stride, name=name + '_0_conv')(x)
        shortcut = tfkl.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x
  
    x = tfkl.Conv2D(filters, kernel_size, 
                    strides=stride, padding='same',
                    kernel_initializer='he_normal',
                    name=name + '_1_conv')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x = tfkl.Activation('relu', name=name + '_1_relu')(x)

    x = tfkl.Conv2D(filters, kernel_size,
                    padding='same',
                    kernel_initializer='he_normal',
                    name=name + '_2_conv')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)

    x = tfkl.Add(name=name + '_add')([x, shortcut])
    x = tfkl.Activation('relu', name=name + '_out')(x)

    return x

def bottleneck(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """Bottleneck layer according to original implementation.
    Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
    Returns:
    Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = tfkl.Conv2D(4 * filters, 1,
                                kernel_initializer='he_normal',
                                strides=stride, name=name + '_0_conv')(x)
        shortcut = tfkl.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = tfkl.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = tfkl.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tfkl.Activation('relu', name=name + '_1_relu')(x)

    x = tfkl.Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tfkl.Activation('relu', name=name + '_2_relu')(x)

    x = tfkl.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tfkl.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tfkl.Add(name=name + '_add')([shortcut, x])
    x = tfkl.Activation('relu', name=name + '_out')(x)
    return x 

def basic_stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked basic conv blocks.
    Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.
    Returns:
    Output tensor for the stacked blocks.
    """
    x = basic_block(x, filters, stride=stride1, name=name + '_block1')
    
    for i in range(2, blocks + 1):
        x = basic_block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def bottleneck_stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked bottleneck blocks.
    Arguments:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.
    Returns:
    Output tensor for the stacked blocks.
    """
    x = bottleneck(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = bottleneck(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x

def ResNet(input_shape,
           stack_fn,
           name):

    height, width, num_channels = input_shape
    img_input = tfkl.Input(shape=input_shape)

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if height <= 32 or width <= 32:
        # CIFAR stem
        x =  tfkl.ZeroPadding2D(padding=1,
                                name='conv1_pad')(img_input)
        x = tfkl.Conv2D(64, 3,
                        strides=1,
                        kernel_initializer='he_normal',
                        name='conv1_conv')(x)
        x = tfkl.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = tfkl.Activation('relu', name='conv1_relu')(x)
    else:
        x = tfkl.ZeroPadding2D(padding=3,
                                name='conv1_pad')(img_input)
        x = tfkl.Conv2D(64, 7,
                        strides=2,
                        kernel_initializer='he_normal',
                        name='conv1_conv')(x)
        x = tfkl.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = tfkl.Activation('relu')(x)
        x = tfkl.ZeroPadding2D(padding=1, name='pool1_pad')(x)
        x = tfkl.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    model = tf.keras.Model(img_input, x, name=name)

    return model

def ResNet18(input_shape,
             name='resnet18'):
  
    def stack_fn(x):
        x = basic_stack(x, 64, 2, stride1=1, name='conv2')
        x = basic_stack(x, 128, 2, name='conv3')
        x = basic_stack(x, 256, 2, name='conv4')
        return basic_stack(x, 512, 2, name='conv5')

    return ResNet(input_shape, stack_fn, name)

def ResNet34(input_shape,
             name='resnet18'):
  
    def stack_fn(x):
        x = basic_stack(x, 64, 3, stride1=1, name='conv2')
        x = basic_stack(x, 128, 4, name='conv3')
        x = basic_stack(x, 256, 6, name='conv4')
        return basic_stack(x, 512, 3, name='conv5')

    return ResNet(input_shape, stack_fn, name)

def ResNet50(input_shape,
             name='resnet50'):
  
    def stack_fn(x):
        x = bottleneck_stack(x, 64, 3, stride1=1, name='conv2')
        x = bottleneck_stack(x, 128, 4, name='conv3')
        x = bottleneck_stack(x, 256, 6, name='conv4')
        return bottleneck_stack(x, 512, 3, name='conv5')

    return ResNet(input_shape, stack_fn, name)