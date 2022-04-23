'''
Keras model for image denoising
'''

from typing import Tuple
import tensorflow as tf


class UNet2D:
    '''
    2d unet, use addition rather than concatenating in the skip path
    '''
    def __init__(
        self,
        input_shape: Tuple[int] = (256, 256, 1),
        output_channel: int = None,
        nconv_per_module: int = 2,
        down_features: Tuple[int] = (64, 64),
        bottleneck_features: int = 64,
        up_features: Tuple[int] = None,
        strides: Tuple[int] = (1, 1),
        use_adding: bool = False,
        use_bn: bool = False,
        use_bilinear_upsampling: bool = False,
        use_relu_output: bool = False,
        dropout_rate: float = 0.0,
        lrelu: float = 0.2,
        use_residual: bool = False,
        name=None
    ):

        if output_channel is None:
            output_channel = input_shape[-1]
        if up_features is None:
            up_features = down_features[::-1]

        self.input_shape = input_shape
        self.output_channel = output_channel
        self.nconv_per_module = nconv_per_module
        self.down_features = down_features
        self.up_features = up_features
        self.bottleneck_features = bottleneck_features
        self.strides = strides
        self.use_adding = use_adding
        self.use_bn = use_bn
        self.use_bilinear_upsampling = use_bilinear_upsampling
        self.use_relu_output = use_relu_output
        self.dropout_rate = dropout_rate
        self.lrelu = lrelu
        self.use_residual = use_residual
        self.name = name

        assert(len(down_features) == len(up_features) and len(strides) == len(up_features))

    def normalization(self, x):
        if self.use_bn:
            return tf.keras.layers.BatchNormalization(scale=False)(x)
        else:
            return x

    def build(self):
        inputs = tf.keras.Input(shape=self.input_shape, name='input')
        x = inputs

        # downsample modules
        down_outputs = []
        for i in range(len(self.down_features)):
            for k in range(self.nconv_per_module):
                x = tf.keras.layers.Conv2D(self.down_features[i], 3, padding='same')(x)
                x = self.normalization(x)
                x = tf.keras.layers.LeakyReLU(alpha=self.lrelu)(x)
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            down_outputs.append(x)
            # down sample
            x = tf.keras.layers.Conv2D(self.down_features[i], 3, self.strides[i], padding='same')(x)
            x = self.normalization(x)
            x = tf.keras.layers.LeakyReLU(alpha=self.lrelu)(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # bottleneck module
        for k in range(self.nconv_per_module):
            x = tf.keras.layers.Conv2D(self.down_features[i], 3, padding='same')(x)
            x = self.normalization(x)
            x = tf.keras.layers.LeakyReLU(alpha=self.lrelu)(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # upsample modules
        for i in range(len(self.up_features)):
            # correpsonding downsampling module
            idown = len(self.up_features) - i - 1
            # upsample
            if self.use_bilinear_upsampling:
                x = tf.keras.layers.Conv2D(self.up_features[i], 3, padding='same')(x)
                x = tf.keras.layers.UpSampling2D(size=self.strides[idown], interpolation='bilinear')(x)
            else:
                x = tf.keras.layers.Conv2DTranspose(self.up_features[i], 3, self.strides[idown], padding='same')(x)
            x = self.normalization(x)
            x = tf.keras.layers.LeakyReLU(alpha=self.lrelu)(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            # combine with downsampling layer
            if self.use_adding:
                x = x + down_outputs[idown]
            else:
                x = tf.keras.layers.concatenate([x, down_outputs[idown]])
            # convolution modules
            for k in range(self.nconv_per_module):
                x = tf.keras.layers.Conv2D(self.up_features[i], 3, padding='same')(x)
                x = self.normalization(x)
                x = tf.keras.layers.LeakyReLU(alpha=self.lrelu)(x)
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # output layer
        x = tf.keras.layers.Conv2D(self.output_channel, 1, padding='same')(x)

        # relu for variance estimation
        if self.use_relu_output:
            x = tf.keras.layers.ReLU()(x)

        if self.use_residual:
            x = inputs + x

        self.model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)

        return self.model
