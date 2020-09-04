""" 3D/2D compatible tf keras Resnet implementation of 
Raghavendra Kotikalapudi's raghakot/keras-resnet
and Cheng-Kun Yang's jimmy15923/3D_ResNet.py """


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model


def set_dimensionality(use_2d):
    """ Custom import tf keras function in their appropriate dimensionality """
    global CONV_VECTOR  # use as tuple for kernel size, strides, etc. in appropriate Dim
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS

    global MaxPooling
    global Conv
    global AveragePooling

    if use_2d:
        from tensorflow.keras.layers import MaxPooling2D as MaxPooling
        from tensorflow.keras.layers import Conv2D as Conv
        from tensorflow.keras.layers import AveragePooling2D as AveragePooling
        CONV_VECTOR = np.array([1, 1])
    else:
        from tensorflow.keras.layers import MaxPooling3D as MaxPooling
        from tensorflow.keras.layers import Conv3D as Conv
        from tensorflow.keras.layers import AveragePooling3D as AveragePooling
        CONV_VECTOR = np.array([1, 1, 1])

    if tf.keras.backend.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = None if use_2d else 3
        CHANNEL_AXIS = 3 if use_2d else 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = None if use_2d else 4


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = tf.keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(input) 
    return tf.keras.layers.Activation("relu")(norm) #Activation("relu")(norm)


def _conv_bn_relu(**conv_params):  #filters,
#                   kernel_size,
#                   strides=tuple(1 * CONV_VECTOR),
#                   kernel_initializer="he_normal",
#                   padding="same",
#                   kernel_regularizer=tf.keras.regularizers.l2(1e-8)):  
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", tuple(1 * CONV_VECTOR))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                tf.keras.regularizers.l2(1e-8))
    def f(input):
        conv = Conv(filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer=kernel_initializer,
                    padding=padding,
                    kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)
    return f


def _get_block(identifier):
    """ Choose either bottleneck or basic type block function, input as
    'basic_block' or 'bottleneck_block' """
    if type(identifier) == str:
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def _shortcut(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = input.shape[DIM1_AXIS] // residual.shape[DIM1_AXIS]
    stride_dim2 = input.shape[DIM2_AXIS] // residual.shape[DIM2_AXIS]
    if len(input.shape) >= 3:
        stride_dim3 = input.shape[DIM3_AXIS] // residual.shape[DIM3_AXIS]
        strides = (stride_dim1, stride_dim2, stride_dim3)
    else:
        strides = (stride_dim1, stride_dim2)
    # Trim strides down to correct dimensionality, using the conv vector as a proxy 
    # strides = tuple[strides[i] for i in range(len(CONV_VECTOR))]

    equal_channels = residual.shape[CHANNEL_AXIS] == input.shape[CHANNEL_AXIS]
    
    shortcut = input
    print(shortcut)
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv(filters=residual.shape[CHANNEL_AXIS],
                        kernel_size=tuple(1 * CONV_VECTOR),
                        strides=strides,
                        kernel_initializer="he_normal", padding="valid",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
                        )(input)
    print(shortcut)
    return tf.keras.layers.add([shortcut, residual])


def basic_block(filters, strides=None, kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D
    implementation by jimmy15923."""
    if strides is None:
        strides = tuple(1 * CONV_VECTOR)

    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv(filters=filters,
                         kernel_size=tuple(3 * CONV_VECTOR),
                         strides=strides,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer
                         )(input)
        else:
            conv1 = _conv_bn_relu(filters=filters,
                                  kernel_size=tuple(3 * CONV_VECTOR),
                                  strides=strides,
                                  kernel_regularizer=kernel_regularizer
                                  )(input)

        residual = _conv_bn_relu(filters=filters,
                                 kernel_size=tuple(3 * CONV_VECTOR),
                                 kernel_regularizer=kernel_regularizer
                                 )(conv1)

        return _shortcut(input, residual)

    return f


def bottleneck_block(filters, 
                     strides=None,
                     kernel_regularizer=tf.keras.regularizers.l2(1e-8),
                     is_first_block_of_first_layer=False):
    """Basic 3 X 3 or 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl by jimmy15923."""
    if strides is None:
        strides = tuple(1 * CONV_VECTOR)

    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv(filters=filters, kernel_size=tuple(1 * CONV_VECTOR),
                            strides=strides, padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=kernel_regularizer
                            )(input)
        else:
            conv_1_1 = _conv_bn_relu(filters=filters, kernel_size=tuple(1 * CONV_VECTOR),
                                     strides=strides,
                                     kernel_regularizer=kernel_regularizer
                                     )(input)

        conv_3_3 = _conv_bn_relu(filters=filters, kernel_size=tuple(3 * CONV_VECTOR),
                                 kernel_regularizer=kernel_regularizer
                                 )(conv_1_1)
        residual = _conv_bn_relu(filters=filters * 4, kernel_size=tuple(1 * CONV_VECTOR),
                                 kernel_regularizer=kernel_regularizer
                                 )(conv_3_3)

        return _shortcut(input, residual)

    return f


def _residual_block(block_function, filters, kernel_regularizer, repetitions,
                    is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = tuple(1 * CONV_VECTOR)
            if i == 0 and not is_first_layer:
                strides = tuple(2 * CONV_VECTOR)

            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


class Bottleneck(tf.keras.Sequential):
    """ TODO: Finish and compare with https://github.com/facebookresearch/swav/blob/master/src/resnet50.py """
    def __init__(self):
        super(Basemodel, self).__init__()
        norm_layer = tf.keras.layers.BatchNormalization()
        
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = tf.keras.activations.relu()
        self.downsample = downsample
        self.stride = stride


class ResnetBuilder(object):
    """Reimplementation of jimmy15923 ResNet3D.
    https://gist.github.com/jimmy15923/9c05b2064bc6de462d21df6285164026 """

    @staticmethod
    def build(
        input_shape, 
        num_outputs, 
        block_fn, 
        repetitions, 
        reg_factor,
        batch_size=None):
        """Instantiate a vanilla ResNet3D keras model.
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottleneck_block'}
            repetitions: Repetitions of unit blocks
            reg_factor: layer regularization penalty
            batch_size: Number of samples processed at a time
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        use_2d = False
        set_dimensionality(use_2d)
        # if len(input_shape) != 4:
        #     raise ValueError("Input shape should be a tuple "
        #                      "(conv_dim1, conv_dim2, conv_dim3, channels) "
        #                      "for tensorflow as backend or "
        #                      "(channels, conv_dim1, conv_dim2, conv_dim3) "
        #                      "for theano as backend")

        block_fn = _get_block(block_fn)
        input = tf.keras.Input(shape=input_shape, batch_size=batch_size)  # tf.keras.layers.Input(shape=input_shape)

        # first conv
        conv1 = _conv_bn_relu(filters=32,
                              kernel_size=tuple(3 * CONV_VECTOR),
                              padding='same',
                              strides=tuple(1 * CONV_VECTOR),
                              kernel_regularizer=tf.keras.regularizers.l2(reg_factor)
                              )(input)
        pool1 = MaxPooling(strides=tuple(2 * CONV_VECTOR),
                           padding="same")(conv1)
        block = pool1

        # repeat blocks
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn,
                                    filters=filters,
                                    kernel_regularizer=tf.keras.regularizers.l2(reg_factor),
                                    repetitions=r, is_first_layer=(i == 0)
                                    )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average pool and classification
        # pool_shape = data[0].element_spec.shape
        pool2 = AveragePooling(pool_size=(block.shape[DIM1_AXIS], block.shape[DIM2_AXIS], block.shape[DIM3_AXIS]),
                               strides=tuple(1 * CONV_VECTOR))(block_output)
        flatten1 = tf.keras.layers.Flatten()(pool2)
        
        #flatten1 = GlobalAveragePooling3D()(block_output)
        # if num_outputs > 1:
        #     dense = tf.keras.layers.Dense(
        #                 units=num_outputs,
        #                 kernel_initializer="he_normal",
        #                 activation="softmax",
        #                 kernel_regularizer=tf.keras.regularizers.l2(reg_factor))(flatten1)
        # else:
        #     dense = tf.keras.layers.Dense(
        #                 units=num_outputs,
        #                 kernel_initializer="he_normal",
        #                 activation="sigmoid",
        #                 kernel_regularizer=tf.keras.regularizers.l2(reg_factor))(flatten1)

        dense = tf.keras.layers.Dense(
                        units=num_outputs,
                        kernel_initializer="he_normal",
                        activation="sigmoid",
                        kernel_regularizer=tf.keras.regularizers.l2(reg_factor))(flatten1)

        # conv layer outputting same dim but filter number
        output == conv1x1
        num_classes == num filters  Conv layer
        softmax

        model = tf.keras.models.Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, batch_size=None, reg_factor=1e-4):
        """Build resnet 18."""
        return ResnetBuilder.build(input_shape, num_outputs, basic_block,
                                repetitions=[2, 2, 2, 2], reg_factor=reg_factor, 
                                batch_size=batch_size)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, batch_size=None, reg_factor=1e-4):
        """Build resnet 34."""
        return ResnetBuilder.build(input_shape, num_outputs, basic_block,
                                repetitions=[3, 4, 6, 3], reg_factor=reg_factor, 
                                batch_size=batch_size)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, batch_size=None, reg_factor=1e-4):
        """Build resnet 50."""
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck_block,
                                repetitions=[3, 4, 6, 3], reg_factor=reg_factor, 
                                batch_size=batch_size)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, batch_size=None, reg_factor=1e-4):
        """Build resnet 101."""
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck_block,
                                repetitions=[3, 4, 23, 3], reg_factor=reg_factor, 
                                batch_size=batch_size)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, batch_size=None, reg_factor=1e-4):
        """Build resnet 152."""
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck_block,
                                repetitions=[3, 8, 36, 3], reg_factor=reg_factor, 
                                batch_size=batch_size)


class Basemodel(tf.keras.Model):
    """ ResNet50 in 2D or 3D """
    def __init__(self):
        super(Basemodel, self).__init__()

        use_2d = True

        self.model = ResNet50()

    def call(self):
        # specify backbone
        self.skip_connections = []

        self.blocks = 0 

        self.backbone = Model(self.model)


if __name__ == "__main__":

    # generate fake data
    x = np.random.randint(0, 8, size=(100, 8, 32, 32, 2))
    y = np.random.randint(0, 8, size=(100, 8, 32, 32, 2))

    # y = np.random.choice([0, 1], size=(1000,))
    y = tf.keras.utils.to_categorical(y, 8)

    # build model  and copile it
    res_model = ResnetBuilder.build_resnet_18(input_shape=(8, 32, 32, 2), num_outputs=8)

    res_model.compile(tf.keras.optimizers.Adam(), loss="categorical_crossentropy")

    res_model.summary()

    # train ResMNet 3D model
    res_model.fit(x, y, batch_size=16, epochs=3)
