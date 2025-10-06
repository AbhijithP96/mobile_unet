import pytest
import numpy as np
import tensorflow as tf
import keras

from src.nn.encoder import Encoder

# ----------------------------------------
#           MobileNet Encoder
# ----------------------------------------


def test_encoder_initialization():
    """
    Test that encoder can be created without errors.
    """

    encoder = Encoder()

    assert encoder is not None
    assert encoder.size == (
        224,
        224,
        3,
    ), f"Encoder size mismatch, Expected (224, 224, 3) but got {encoder.size}"


def test_encoder_backbone():
    """
    Test that encoder has a MobileNetV2 backbone.
    """
    encoder = Encoder()

    assert hasattr(encoder, "base_model")
    assert encoder.base_model is not None

    assert isinstance(encoder.base_model, keras.Model)
    assert encoder.base_model.input.shape == (
        None,
        224,
        224,
        3,
    ), f"Input shape mismatch, Expected (None, 224, 224, 3) but got {encoder.base_model.input.shape}"

    # compare with the mobilenetv2 architecture
    mobilenet = keras.applications.MobileNetV2(weights=None, include_top=False)

    mobilenet_layers = [layer.name for layer in mobilenet.layers]
    encoder_layers = [layer.name for layer in encoder.base_model.layers]

    # checking architecture exluding the input layer
    assert all(
        name in encoder_layers for name in mobilenet_layers[1:]
    ), f"MobileNetV2 backbone expected but got otherwise."


def test_encoder_forward_pass():
    """Test basic forward pass through encoder."""
    encoder = Encoder()

    # Create dummy input
    test_input = tf.random.normal((1, 224, 224, 3))

    # Forward pass
    outputs = encoder(test_input)

    # Should return a list of outputs
    assert isinstance(outputs, list)
    assert (
        len(outputs) == 5
    ), f"Encoder output size is wrong, Expected 5 but got {len(outputs)}"  # bottleneck + 4 skip connections

    expected_shapes = [
        (1, 7, 7, 1280),
        (1, 14, 14, 576),
        (1, 28, 28, 192),
        (1, 56, 56, 144),
        (1, 112, 112, 96),
    ]

    for i, output in enumerate(outputs):
        assert len(output.shape) == 4, f"Output at index {i} is not 4D"
        assert (
            output.shape == expected_shapes[i]
        ), f"Output shape mismatch at index {i}, Expected {expected_shapes[i]} but got {output.shape}"


def test_encoder_reproducible():
    """Test that encoder gives same output for same input."""
    encoder = Encoder()

    test_input = tf.ones((1, 224, 224, 3))

    # Two forward passes with same input
    output1 = encoder(test_input)
    output2 = encoder(test_input)

    # Should be identical
    for i in range(len(output1)):
        np.testing.assert_array_equal(output1[i].numpy(), output2[i].numpy())


def test_encoder_outputs_are_positive():
    """Test that encoder outputs are positive (ReLU activations)."""
    encoder = Encoder()

    test_input = tf.random.uniform((1, 224, 224, 3), 0, 1)
    outputs = encoder(test_input)

    # All outputs should be >= 0 (ReLU activations)
    for output in outputs:
        assert tf.reduce_min(output) >= 0, "Encoder output contains negative values."
