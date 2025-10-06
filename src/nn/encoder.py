import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, size: tuple = (224, 224, 3), **kwargs):
        """
        Pre-trained MobileNetV2 Encoder for U-Net.

        Args:
            size (tuple): Input image size (H, W).
        """
        super().__init__(**kwargs)

        self.size = size

        self._build_layers()

    def _build_layers(self):
        """
        Constructs the encoder and the skip connection for the decoder.
        """

        # Build the base MobileNetV2 backbone
        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=self.size
        )
        self.base_model.trainable = False

        # Define skip connection layer names
        self.skip_layer_names = [
            "block_13_expand_relu",  # 14x14
            "block_6_expand_relu",  # 28x28
            "block_3_expand_relu",  # 56x56
            "block_1_expand_relu",  # 112x112
        ]

        # Collect outputs
        skip_outputs = [
            self.base_model.get_layer(name).output for name in self.skip_layer_names
        ]
        encoder_output = self.base_model.output

        # Build encoder model
        self.encoder_model = tf.keras.models.Model(
            inputs=self.base_model.input, outputs=[encoder_output] + skip_outputs
        )

    def call(self, inputs, training=False):
        """
        Forward pass through the encoder.
        Returns:
            List[tf.Tensor]: [bottleneck, skip4, skip3, skip2, skip1]
        """
        return self.encoder_model(inputs, training=training)

    @property
    def output_shapes(self):
        """Return shapes of encoder outputs for debugging/decoder design."""
        return [out.shape for out in self.encoder_model.outputs]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":

    encoder = Encoder(size=(224, 224, 3))

    # print output shape
    print(encoder.output_shapes)
