import pytest
import numpy as np
import tensorflow as tf
import keras

from src.nn.decoder import MobileDecoder

def test_decoder_initialization():
    """Test that decoder can be created without errors."""
    filter_list = [128, 64, 32, 16]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    assert decoder is not None
    assert decoder.filter_list == filter_list
    assert decoder.num_classes == 1


def test_decoder_has_required_layers():
    """Test that decoder has all required layer components."""
    filter_list = [128, 64, 32, 16]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    # Check all layer lists exist
    assert hasattr(decoder, 'upconv')
    assert hasattr(decoder, 'match')
    assert hasattr(decoder, 'conv1')
    assert hasattr(decoder, 'conv2')
    assert hasattr(decoder, 'last_upconv')
    assert hasattr(decoder, 'output_layer')
    
    # Check correct number of layers
    assert len(decoder.upconv) == len(filter_list)
    assert len(decoder.match) == len(filter_list)
    assert len(decoder.conv1) == len(filter_list)
    assert len(decoder.conv2) == len(filter_list)


def test_decoder_binary_classification():
    """Test decoder with binary classification (num_classes=1)."""
    filter_list = [128, 64, 32, 16]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    # Output layer should have sigmoid activation for binary
    assert decoder.output_layer.activation.__name__ == 'sigmoid'
    assert decoder.output_layer.filters == 1

def test_decoder_forward_pass():
    """Test basic forward pass through decoder."""
    filter_list = [128, 64, 32, 16]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    # Create mock encoder outputs (bottleneck + 4 skip connections)
    # Simulating typical encoder output shapes from your encoder
    bottleneck = tf.random.normal((1, 7, 7, 1280))     # 7x7 bottleneck
    skip4 = tf.random.normal((1, 14, 14, 576))         # 14x14
    skip3 = tf.random.normal((1, 28, 28, 192))         # 28x28  
    skip2 = tf.random.normal((1, 56, 56, 144))         # 56x56
    skip1 = tf.random.normal((1, 112, 112, 96))        # 112x112
    
    encoder_outputs = [bottleneck, skip4, skip3, skip2, skip1]
    
    # Forward pass
    output = decoder(encoder_outputs)
    
    # Should return single output tensor
    assert isinstance(output, tf.Tensor)
    assert len(output.shape) == 4  # 4D tensor (B, H, W, C)

def test_decoder_output_shape():
    """Test that decoder produces correct output shape."""
    filter_list = [128, 64, 32, 16]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    batch_size = 2
    # Create encoder outputs with batch size 2
    bottleneck = tf.random.normal((batch_size, 7, 7, 1280))
    skip4 = tf.random.normal((batch_size, 14, 14, 576))
    skip3 = tf.random.normal((batch_size, 28, 28, 192))
    skip2 = tf.random.normal((batch_size, 56, 56, 144))
    skip1 = tf.random.normal((batch_size, 112, 112, 96))
    
    encoder_outputs = [bottleneck, skip4, skip3, skip2, skip1]
    
    output = decoder(encoder_outputs)
    
    # Output should be upsampled to original size
    expected_shape = (batch_size, 224, 224, 1)  # 112 * 2 = 224 (last upsampling)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # Values should be in [0, 1] due to sigmoid activation
    assert tf.reduce_min(output) >= 0.0 and tf.reduce_max(output) <= 1.0, "Output values not in [0, 1] range"

def test_decoder_different_filter_sizes():
    """Test decoder with different filter sizes."""
    filter_list = [
        (128, 64, 32, 16),
        (256, 128, 64, 32),
        (512, 256, 128, 64)
    ]

    # create encoder outputs for testing
    batch_size = 2
    # Create encoder outputs with batch size 2
    bottleneck = tf.random.normal((batch_size, 7, 7, 1280))
    skip4 = tf.random.normal((batch_size, 14, 14, 576))
    skip3 = tf.random.normal((batch_size, 28, 28, 192))
    skip2 = tf.random.normal((batch_size, 56, 56, 144))
    skip1 = tf.random.normal((batch_size, 112, 112, 96))

    
    for filters in filter_list:
        decoder = MobileDecoder(filter_list=filters, num_classes=1)
        assert decoder.filter_list == filters

        encoder_outputs = [bottleneck, skip4, skip3, skip2, skip1]
        output = decoder(encoder_outputs)

        expected_shape = (batch_size, 224, 224, 1)
        assert output is not None, "Decoder output is None"
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
        assert tf.reduce_min(output) >= 0.0 and tf.reduce_max(output) <= 1.0, "Output values not in [0, 1] range"

        del decoder  # Clean up

def test_decoder_reproducible():
    """Test that decoder gives same output for same input."""
    filter_list = [64, 32]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    # Fixed inputs
    bottleneck = tf.ones((1, 7, 7, 1280))
    skip1 = tf.ones((1, 14, 14, 576))
    skip2 = tf.ones((1, 28, 28, 192))
    
    encoder_outputs = [bottleneck, skip1, skip2]
    
    # Two forward passes with same input
    output1 = decoder(encoder_outputs)
    output2 = decoder(encoder_outputs)
    
    # Should be identical
    np.testing.assert_array_equal(output1.numpy(), output2.numpy())

def test_decoder_trainable_parameters():
    """Test that decoder has trainable parameters."""
    filter_list = [128, 64, 32, 16]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    # Build decoder by running a forward pass
    bottleneck = tf.random.normal((1, 7, 7, 1280))
    skip1 = tf.random.normal((1, 14, 14, 576))
    skip2 = tf.random.normal((1, 28, 28, 192))
    skip3 = tf.random.normal((1, 56, 56, 144))
    skip4 = tf.random.normal((1, 112, 112, 96))
    
    encoder_outputs = [bottleneck, skip1, skip2, skip3, skip4]
    _ = decoder(encoder_outputs)
    
    # Should have trainable parameters
    trainable_params = sum([tf.size(var).numpy() for var in decoder.trainable_variables])
    assert trainable_params > 0


def test_decoder_gradient_flow():
    """Test that gradients flow through decoder."""
    filter_list = [64, 32]
    decoder = MobileDecoder(filter_list=filter_list, num_classes=1)
    
    # Create encoder outputs as variables
    bottleneck = tf.Variable(tf.random.normal((1, 7, 7, 1280)))
    skip1 = tf.Variable(tf.random.normal((1, 14, 14, 576)))
    skip2 = tf.Variable(tf.random.normal((1, 28, 28, 192)))
    
    encoder_outputs = [bottleneck, skip1, skip2]
    
    with tf.GradientTape() as tape:
        output = decoder(encoder_outputs)
        loss = tf.reduce_mean(output)
    
    # Should have gradients w.r.t decoder parameters
    decoder_gradients = tape.gradient(loss, decoder.trainable_variables)
    
    # Should have some non-None gradients
    non_none_gradients = [g for g in decoder_gradients if g is not None]
    assert len(non_none_gradients) > 0