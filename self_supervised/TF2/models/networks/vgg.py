import tensorflow as tf
import tensorflow.keras.layers as tfkl


def conv_block(x, filters, num_conv, stride=1, out_name=None):

    for i in range(num_conv):
        x = tfkl.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tfkl.BatchNormalization(axis=-1, momentum=0.95, epsilon=0.001)(x)
        if i == num_conv - 1:
            out_name = out_name
            x = tfkl.ReLU(name=out_name)(x)
        else:
            x = tfkl.ReLU()(x)

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

    x = tfkl.Input(shape=input_shape)
    x1 = conv_block(x, 64, 2)
    x1_pool = tfkl.MaxPooling2D()(x1)
    x2 = conv_block(x1_pool, 128, 2)
    x2_pool = tfkl.MaxPooling2D()(x2)
    x3 = conv_block(x2_pool, 256, 2)
    x3_pool = tfkl.MaxPooling2D()(x3)
    x4 = conv_block(x3_pool, 512, 2)
    x4_pool = tfkl.MaxPooling2D()(x4)
    x5 = conv_block(x4_pool, 1024, 2)

    model = tf.keras.Model(x, [x1, x2, x3, x4, x5])
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


class UpConv(tf.keras.Model):
    def __init__(self,
                 filters,
                 stride,
                 use_transpose=False,
                 name='up_conv'):
        super(UpConv, self).__init__(name=name)

        if use_transpose:
            self.upconv_layer = tfkl.Conv2DTranspose(
                filters, 2, strides=stride
            )
        else:
            self.upconv_layer = tfkl.UpSampling2D(
                stride
            )

        self.conv_1 = tfkl.Conv2D(
            filters, 2, padding='same', use_bias=False
        )
        self.bn_1 = tfkl.BatchNormalization(
            axis=-1, momentum=0.95, epsilon=0.001)
        self.relu_1 = tfkl.ReLU()

        self.conv_2 = tfkl.Conv2D(
            filters, 3, padding='same', use_bias=False
        )
        self.bn_2 = tfkl.BatchNormalization(
            axis=-1, momentum=0.95, epsilon=0.001)
        self.relu_2 = tfkl.ReLU()

        self.conv_3 = tfkl.Conv2D(
            filters, 3, padding='same', use_bias=False
        )
        self.bn_3 = tfkl.BatchNormalization(
            axis=-1, momentum=0.95, epsilon=0.001)
        self.relu_3 = tfkl.ReLU()

    def call(self, inputs, bridge, training=False):

        up = self.upconv_layer(inputs)
        up = self.conv_1(up)
        up = self.bn_1(up, training=training)
        up = self.relu_1(up)

        out = tfkl.concatenate([up, bridge], axis=-1)
        out = self.conv_2(out)
        out = self.bn_2(out, training=training)
        out = self.relu_2(out)

        out = self.conv_3(out)
        out = self.bn_3(out, training=training)
        out = self.relu_3(out)

        return out


class VGG_UNet_Decoder(tf.keras.Model):

    def __init__(self,
                 num_classes,
                 output_activation='softmax',
                 use_transpose=True,
                 name='vgg_unet_decoder'):
        super(VGG_UNet_Decoder, self).__init__(name=name)

        self.conv_list = []
        for i in range(4):
            self.conv_list.append(UpConv(
                filters=64 * (2**i), stride=2
            ))

        self.conv_1x1 = tfkl.Conv2D(
            num_classes, 1, activation=output_activation, padding='same')

    def call(self, x, bridge_list, training=False):

        for i in reversed(range(4)):
            x = self.conv_list[i](x, bridge_list[i], training=training)

        output = self.conv_1x1(x)
        return output
