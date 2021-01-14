import tensorflow as tf
import tensorflow.keras.layers as tfkl

def conv_block(x, filters, num_conv, stride=1):

    bn_axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    for _ in range(num_conv):
        x = tfkl.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tfkl.BatchNormalization(axis=bn_axis, momentum=0.95, epsilon=0.001)(x)
        x = tfkl.ReLU()(x)

    return x

def upconv_block(x, bridge, filters, stride=2, use_transpose=True):

    bn_axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if use_transpose:
        x = tfkl.Conv2DTranspose(filters, 3, padding='same', strides=stride)(x)
    else:
        x = tfkl.UpSampling2D(size=stride)(x)

    x = conv_block(x, filters, 1)
    x = tfkl.concatenate([x, bridge], axis=bn_axis)
    x = conv_block(x, filters, 2)

    return x

def VGG(input_shape,
        stack_fn,
        name):

    img_input = tfkl.Input(shape=input_shape)
    x = stack_fn(img_input)
    model = tf.keras.Model(img_input, x, name=name)
    return model

def VGG_UNet_Encoder(input_shape,
                     name='vgg_unet_encoder'):

    def stack_fn(x):
        for i in range(5):
            x = conv_block(x, 64 * (2**i), 2)
            if i != 4:
                x = tfkl.MaxPooling2D()(x)
        return x

    model = VGG(input_shape, stack_fn, name)
    return model

def VGG16(input_shape,
          name='vgg16'):

    def stack_fn(x):
        x = conv_block(x, 64, 2)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 128, 2)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 256, 2)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 512, 3)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 512, 3)
        return x

    model = VGG(input_shape, stack_fn, name)
    return model

def VGG19(input_shape,
          name='vgg19'):

    def stack_fn(x):
        x = conv_block(x, 64, 2)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 128, 2)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 256, 2)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 512, 4)
        x = tfkl.MaxPooling2D()(x)
        x = conv_block(x, 512, 4)
        return x

    model = VGG(input_shape, stack_fn, name)
    return model

def VGG_UNet_Decoder(input_shapes,
                     num_classes,
                     output_activation='softmax',
                     use_transpose=True,
                     name='vgg_unet_decoder'):

    input_list = []
    for i in range(5):
        in_shape = tfkl.Input(shape=input_shapes[i])
        input_list.append(in_shape)

    x = upconv_block(input_list[0], input_list[i+1], 1024 * (2**-(i + 1)), use_transpose=use_transpose)
    for i in range(3):
        x = upconv_block(x, input_list[i+2], 1024 * (2**-(i + 2)), use_transpose=use_transpose)
    x = tfkl.Conv2D(num_classes,
                    1,
                    activation=output_activation,
                    padding='same')(x)

    model = tf.keras.Model(inputs=input_list, outputs=x, name=name)
    return model
