import tensorflow as tf
import tensorflow.keras.layers as tfkl


def conv_block(x, filters, num_conv, kernel_size=3, stride=1):

    for i in range(num_conv):
        x = tfkl.Conv2D(
            filters, kernel_size, padding='same', use_bias=False)(x)
        x = tfkl.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(x)
        x = tfkl.ReLU()(x)

    return x


def conv_block_seq(filters, num_conv, kernel_size=3, stride=1):
    model = tf.keras.Sequential()
    for i in range(num_conv):
        model.add(
            tfkl.Conv2D(filters, kernel_size, padding='same', use_bias=False))
        model.add(
            tfkl.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001))
        model.add(tfkl.ReLU())
    return model


def VGG(input_shape,
        stack_fn,
        name):

    img_input = tfkl.Input(shape=input_shape)
    x = stack_fn(img_input)
    model = tf.keras.Model(img_input, x, name=name)
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
        x = tfkl.MaxPooling2D()(x)
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
        x = tfkl.MaxPooling2D()(x)
        return x

    model = VGG(input_shape, stack_fn, name)
    return model


def SegNet_Decoder():
    model = tf.keras.Sequential([
        tfkl.UpSampling2D(),
        conv_block_seq(512, 3),
        tfkl.UpSampling2D(),
        conv_block_seq(512, 3),
        tfkl.UpSampling2D(),
        conv_block_seq(256, 2),
        tfkl.UpSampling2D(),
        conv_block_seq(128, 2),
        tfkl.UpSampling2D(),
        conv_block_seq(64, 2),
    ])

    return model


def upconv_block(x, bridge, filters, stride, use_transpose=False):
    if use_transpose:
        x = tfkl.Conv2DTranspose(filters, 2, strides=stride, padding='same')(x)
    else:
        x = tfkl.UpSampling2D(size=stride)(x)

    x = conv_block(x, filters, 1, kernel_size=2)
    x = tfkl.concatenate([x, bridge], axis=-1)
    x = conv_block(x, filters, 2)
    return x


def VGG_UNet(input_shape,
             name='unet'):

    def stack_fn(x):
        bridges = []
        for i in range(5):
            x = conv_block(x, 32 * (2**i), 2)
            if i != 4:
                bridges.append(x)
                x = tfkl.MaxPooling2D()(x)

        for j in reversed(range(4)):
            x = upconv_block(x, bridges[j], 32 * (2**(j)), stride=2)

        return x

    model = VGG(input_shape, stack_fn, name)
    return model


def VGG_UNet_Encoder(input_shape,
                     num_conv=32,
                     name='vgg_unet_encoder'):

    x = tfkl.Input(shape=input_shape)
    x1 = conv_block(x, num_conv, 2)
    x1_pool = tfkl.MaxPooling2D()(x1)
    x2 = conv_block(x1_pool, num_conv*2, 2)
    x2_pool = tfkl.MaxPooling2D()(x2)
    x3 = conv_block(x2_pool, num_conv*4, 2)
    x3_pool = tfkl.MaxPooling2D()(x3)
    x4 = conv_block(x3_pool, num_conv*8, 2)
    x4_pool = tfkl.MaxPooling2D()(x4)
    x5 = conv_block(x4_pool, num_conv*16, 2)

    model = tf.keras.Model(x, [x1, x2, x3, x4, x5])
    return model


def VGG_UNet_Decoder(input_shapes,
                     num_conv=32):

    bridge_4 = tf.keras.Input(input_shapes[4])
    bridge_3 = tf.keras.Input(input_shapes[3])
    bridge_2 = tf.keras.Input(input_shapes[2])
    bridge_1 = tf.keras.Input(input_shapes[1])
    bridge_0 = tf.keras.Input(input_shapes[0])

    x1 = upconv_block(bridge_4, bridge_3, num_conv*8, stride=2)
    x2 = upconv_block(x1, bridge_2, num_conv*4, stride=2)
    x3 = upconv_block(x2, bridge_1, num_conv*2, stride=2)
    x4 = upconv_block(x3, bridge_0, num_conv, stride=2)

    model = tf.keras.Model(
        [bridge_0, bridge_1, bridge_2, bridge_3, bridge_4], x4)
    return model


def build_unet(encoder, decoder, num_classes):

    encoder_outputs = encoder.output
    decoder_output = decoder(encoder_outputs)

    if num_classes == 1:
        output_activation = 'sigmoid'
    else:
        output_activation = 'softmax'

    seg_output = tfkl.Conv2D(num_classes,
                             1,
                             padding='same',
                             activation=output_activation)(decoder_output)

    return tf.keras.Model(
        inputs=encoder.input, outputs=seg_output, name='unet')