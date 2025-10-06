import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose
from keras.saving import register_keras_serializable
import keras


@register_keras_serializable()
class Decoder(tf.keras.layers.Layer):

    def __init__(self, filter_list, num_classes=1, **kwargs):
        """
        Standard Decoder for U-Net style segmentation.

        Args:
            filter_list (_type_): List of filters for each decoder blocks.
        """

        super().__init__(**kwargs)
        self.filter_list = filter_list
        self.num_classes = num_classes

        self._build_layers()

    def _build_layers(self):

        self.upconv = []
        self.match = []
        self.conv1 = []
        self.conv2 = []

        for f in self.filter_list:
            self.upconv.append(
                Conv2DTranspose(
                    filters=f,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )
            self.match.append(Conv2D(f, kernel_size=1, padding="same"))

            self.conv1.append(
                Conv2D(filters=f, kernel_size=3, padding="same", activation="relu")
            )
            self.conv2.append(
                Conv2D(filters=f, kernel_size=3, padding="same", activation="relu")
            )

        self.last_upconv = Conv2DTranspose(
            filters=self.filter_list[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )

        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        self.output_layer = Conv2D(
            filters=1, kernel_size=1, padding="same", activation=activation
        )

    def call(self, inputs):
        """
        Forward pass through the decoder.

        Args:
            inputs (_type_): Encoder Outputs
        """

        x, *skips = inputs

        for i, _ in enumerate(self.filter_list):
            x = self.upconv[i](x)
            skip = skips.pop(0)
            skip = self.match[i](skip)
            x = tf.keras.layers.Concatenate()([x, skip])
            x = self.conv1[i](x)
            x = self.conv2[i](x)

        x = self.last_upconv(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_list": self.filter_list,
                "num_classes": self.num_classes,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class MobileDecoder(keras.layers.Layer):

    def __init__(self, filter_list, num_classes=1, **kwargs):
        """
        Lightweight Decoder (dpethwise convolutions) for U-Net style segmentation.

        Args:
            filter_list (_type_): List of filters for each decoder blocks.
        """

        super().__init__(**kwargs)
        self.filter_list = filter_list
        self.num_classes = num_classes

        self._build_layers()

    def _build_layers(self):
        """
        Constructs the decoder layers.
        """

        self.upconv = []
        self.match = []
        self.conv1 = []
        self.conv2 = []

        for f in self.filter_list:
            self.upconv.append(
                Conv2DTranspose(
                    filters=f,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )
            self.match.append(Conv2D(f, kernel_size=1, padding="same"))

            self.conv1.append(self._create_smaller_conv_layers(f))
            self.conv2.append(self._create_smaller_conv_layers(f))

        self.last_upconv = Conv2DTranspose(
            filters=self.filter_list[-1],
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )

        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        self.output_layer = Conv2D(
            filters=1, kernel_size=1, padding="same", activation=activation
        )

    def _create_smaller_conv_layers(self, f):

        # MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
        # https://arxiv.org/pdf/1704.04861

        layers = keras.Sequential(
            [
                keras.layers.DepthwiseConv2D(kernel_size=3, padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(filters=f, kernel_size=1, padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
            ]
        )

        return layers

    def call(self, inputs):
        """
        Forward pass through the decoder.

        Args:
            inputs (_type_): Encoder Outputs
        """

        x, *skips = inputs

        for i, _ in enumerate(self.filter_list):
            x = self.upconv[i](x)
            skip = skips.pop(0)
            skip = self.match[i](skip)
            x = tf.keras.layers.Concatenate()([x, skip])
            x = self.conv1[i](x)
            x = self.conv2[i](x)

        x = self.last_upconv(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_list": self.filter_list,
                "num_classes": self.num_classes,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":

    from .encoder import Encoder

    encoder = Encoder(size=(224, 224))
    decoder = Decoder([512, 256, 128, 64])

    dummy_input = tf.random.normal((1, 224, 224, 3))
    encoder_outputs = encoder(dummy_input)

    decoder_output = decoder(encoder_outputs)

    print(decoder_output.shape)
