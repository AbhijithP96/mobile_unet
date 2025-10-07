"""
Testing Utilities for Lane Segmentation Model

This module provides functions to evaluate a trained lane segmentation model
that has been logged and versioned in MLflow. The functions automatically
download the specified model from the MLflow Model Registry before running
batch or single-image inference.
"""

from src.data_engine import preprocess, dataset
from src.nn.unet import UNet
import tensorflow as tf
import mlflow
import numpy as np
import keras
import matplotlib.pyplot as plt

from utils import dice_coef
import utils, time


def test_unet(model_name: str, version):
    """
    Evaluate the model on the test dataset.

    Downloads the specified model version from the MLflow Model Registry,
    runs inference on the full test dataset, and computes performance metrics.
    """

    # prepare the dataset
    paths_df = preprocess.preprocess(split="test", finished=True)

    # create the tf dataset
    data = dataset.TuSimple(paths_df, batch_size=8)
    test_data = data.get_dataset(split="test")

    # get the model from the run_id
    mlflow.set_tracking_uri("http://localhost:8080")

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.tensorflow.load_model(model_uri)

    model.summary()
    batch_score = []

    score = model.evaluate(test_data, return_dict=True)

    return score


def test_single(model_name: str, version: int, image_path: str):
    """
    Run inference on a single image using a model from MLflow.

    Downloads the specified model version from the MLflow Model Registry,
    preprocesses the given image, predicts the segmentation mask, and optionally
    displays or saves the overlaid result.
    """
    # get the model from the run_id
    mlflow.set_tracking_uri("http://localhost:8080")

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.tensorflow.load_model(model_uri)

    model.summary()

    # prepare the image for inference
    start = time.time()
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, tf.convert_to_tensor((224, 224), dtype=tf.int32))
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    image = tf.expand_dims(image, axis=0)

    # predict
    mask = model.predict(image)
    mask = (mask[0] > 0.5) * 255.0

    end = time.time()
    print(f"Time Taken for Inference: {end-start:.2f} seconds")

    return mask


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Test the Mobile UNet model")
    parser.add_argument("--model", type=str, default="mobile_unet", help="Model name")
    parser.add_argument("--version", type=int, default=1, help="Model version")
    args = parser.parse_args()

    name = args.model
    version = args.version

    score = test_unet(name, version)

    print(f"Test Dice Coefficient: {score['dice_coef']:.4f}")
    print(f"Test Loss: {score['loss']:.4f}")
    print(f'Test Precision: {score["precision"]:.4f}')
    print(f'Test Recall: {score["recall"]:.4f}')

    # print(score)
    # utils.dump_json(score, name='test_result')

    # image_path = "./data/20.jpg"

    # mask = test_single(name, version, image_path)

    # show mask
    # plt.imshow(np.uint8(mask))
    # plt.show()
