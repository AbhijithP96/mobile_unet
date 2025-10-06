"""
Inference Module for Lane Segmentation

This module provides a high-level interface for loading a trained Keras
segmentation model and performing inference. The model can be initialized
either from a local path or downloaded from an MLflow model registry,
depending on configuration.

Key Features:
    - Automatically loads and warms up the model to avoid first-inference delay.
    - Provides a simple `predict()` method to generate segmentation masks.
    - Includes preprocessing utilities to prepare images for inference.
"""

import time
import numpy as np
import mlflow
import tensorflow as tf
import cv2
import keras
from src.nn.unet import UNet


class Inference:
    """
    Inference class for lane segmentation models.

    This class loads a pre-trained Keras model (from a local directory or MLflow),
    builds it with a dummy input to perform a warm-up forward pass, and provides
    a convenient interface for running inference on new images.
    """

    def __init__(self, mlflow_model: bool = False, **kwargs):
        """
        Initialize the Inference class.

        Loads a Keras model either from a local directory or from the MLflow model registry,
        depending on the value of `mlflow_model`.

        Args:
            mlflow_model (bool, optional):
                If True, load the model from MLflow registry using `model_name` and `version`.
                If False, load the model from a local directory specified by `model_path`.
                Defaults to False.

        Keyword Args:
            model_name (str): Name of the registered model in MLflow (required if mlflow_model=True).
            version (int): Version of the MLflow model to load (required if mlflow_model=True).
            model_path (str): Local path to the saved Keras model (required if mlflow_model=False).

        Raises:
            ValueError: If the required arguments for the selected mode (MLflow or local)
                        are not provided.
        """

        if mlflow_model:

            mlflow.set_tracking_uri("http://localhost:8080")

            model_name = kwargs.get("model_name", "mobile_unet")
            version = kwargs.get("version", 1)
            model_uri = f"models:/{model_name}/{version}"

            try:
                self.model = mlflow.tensorflow.load_model(model_uri)

            except Exception:
                raise ValueError(
                    "Error Occured while loading model from mlflow \n"
                    "Ensure model_name and version are correctly mentioned and passed while initializing Inference object"
                )

        else:

            model_path = kwargs.get("model_path", "./saved_model/mobile_unet.keras")

            try:

                self.model = keras.models.load_model(model_path, compile=False)

            except Exception as e:
                print(str(e))
                raise ValueError(
                    "Error occured while loading model \n"
                    "Ensure model_path is correct and passed while initializing Inference object"
                )

        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        print("Model loaded and warmed up!!")
        print("=" * 60)

    def preprocess_image(self, image):
        """
        Preprocess an input image for model inference.

        This method resizes, normalizes, and reshapes the image to match
        the model's expected input format.
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224, 224))

        image = image.astype(np.float32) / 127.5 - 1.0

        return np.expand_dims(image, axis=0)

    def predict(self, image):
        """
        Run inference on an input image and return the predicted mask.

        This method first preprocesses the input image, then passes it through
        the model to obtain a segmentation mask.
        """

        start = time.time()

        image = self.preprocess_image(image)
        preprocess_time = time.time() - start

        infer_start = time.time()
        mask = self.model.predict(image, verbose=0)
        infer_time = time.time() - infer_start

        total_time = time.time() - start

        print(f"Preprocessing: {preprocess_time:.3f}s")
        print(f"Inference: {infer_time:.3f}s")
        print(f"Total: {total_time:.3f}s")

        return ((mask[0] > 0.5) * 255.0).astype(np.uint8)


if __name__ == "__main__":
    import sys

    path = sys.argv[1]

    ob = Inference(mlflow_model=False, model_path="./saved_model/mobile_unet.keras")
    # ob = Inference(mlflow_model=True, model_name = 'mobile_unet', version= 1)

    image = cv2.imread(path)
    mask = ob.predict(image)

    # ob.model.save('./saved_model/mobile_unet.keras')
