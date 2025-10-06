"""
CLI Client for Lane Segmentation API

This module provides a simple command-line interface (CLI) for sending images
to the FastAPI inference server and displaying the segmented results.

Usage:
    python predict.py <path_to_image>
"""

import requests
import base64
import cv2
import numpy as np


def get_segmented_image(image_path):
    """
    Send an image to the lane segmentation API and display the result.

    This function reads an image from the given path, encodes it as base64,
    sends it to the specified API endpoint, decodes the returned segmented
    image, and displays it to the user.

    Args:
        image_path (str): Path to the image file to be sent for segmentation.
    """

    file_path = image_path
    url = "http://0.0.0.0:8000/segment"

    with open(file_path, "rb") as f:
        response = requests.post(url, files={"file": f})

    data = response.json()
    seg_b64 = data["segmented_base64"]

    # Decode back to OpenCV image
    seg_bytes = base64.b64decode(seg_b64)
    seg_np = np.frombuffer(seg_bytes, np.uint8)
    seg_img = cv2.imdecode(seg_np, cv2.IMREAD_COLOR)

    cv2.imshow("Predicted Mask", cv2.resize(seg_img, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import sys

    path = sys.argv[1]

    get_segmented_image(path)
