"""
Mobilener-UNet architecture for effiecient lane segmentation

This module implements a UNet architecture using frozen mobilenet encoder backbone
with frozen weights from imagenet and allows the decoder to be either standard unet or
a unet with depthwise convolution in its conv layers to create a lighter version of the architecture.

The decoder can be configured in two modes:
    1. **Standard UNet Decoder** : Uses regular convolutional layers for feature
       upsampling and reconstruction.
    2. **Lightweight Decoder** : Replaces standard convolutions with depthwise
       separable convolutions, resulting in a more parameter-efficient and
       computationally lightweight architecture.
"""

import tensorflow as tf
from keras.saving import register_keras_serializable
from .encoder import Encoder
from .decoder import Decoder, MobileDecoder
import io
import tempfile


@register_keras_serializable()
class UNet(tf.keras.models.Model):
    """
    This class implements a UNet-style architecture using a frozen MobileNet
    encoder as the backbone for feature extraction. The decoder can be configured
    to use either standard convolutions or depthwise separable convolutions to
    reduce computational cost and model size.
    """

    def __init__(
        self,
        input_size: tuple,
        filter_list: list,
        num_classes: int = 1,
        mobile=False,
        **kwargs,
    ):
        """
        High-level interface for the U-Net model with Mobilenet encoder

        Args:
            input_size (tuple): Input image size (H,W)
            filter_list (list): Number of filters for each decoder block
            num_classes (int, optional): Number of output classes. Defaults to 1.
            mobile (bool, optional): Flag to use the lightweight decoder. Defaults to False.
        """

        super().__init__(**kwargs)

        self.input_shape = input_size
        self.filter_list = filter_list
        self.num_classes = num_classes

        self.mobile = mobile

        self._build_layers()

    def _build_layers(self):

        # build the encoder
        self.encoder = Encoder(self.input_shape)

        # build the decoder
        if self.mobile:
            self.decoder = MobileDecoder(self.filter_list, self.num_classes)

        else:
            self.decoder = Decoder(self.filter_list, self.num_classes)

    def call(self, inputs):
        """Forward pass of the U-Net Model"""
        x = self.encoder(inputs)
        x = self.decoder(x)

        return x

    def build(self):
        x = tf.keras.Input(shape=self.input_shape)
        _ = self.call(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_size": self.input_shape,
                "filter_list": self.filter_list,
                "num_classes": self.num_classes,
                "mobile": self.mobile,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_summary_path(self):
        """
        Function to get the model summary artifact for mlflow logger.
        """

        stream = io.StringIO()
        self.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_str = stream.getvalue()
        stream.close()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(summary_str)
            tmp_path = tmp.name

        return tmp_path
