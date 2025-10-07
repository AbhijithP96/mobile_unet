import tensorflow as tf
import keras
from src.nn.unet import UNet

model = keras.models.load_model("./saved_model/mobile_unet.keras", compile=False)

input_shape = (1, 224, 224, 3)
concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec(input_shape, tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

tflite_model = converter.convert()

with open("./saved_model/mobile_unet.tflite", "wb") as f:
    f.write(tflite_model)
