"""
FastAPI Application for Lane Segmentation Inference

This module initializes and configures a FastAPI application that serves
a trained lane segmentation model through REST endpoints.

Endpoints:
    GET  /       - Health check / welcome endpoint.
    POST /segment - Accepts an image file, runs inference, and returns
                    a segmented image with lane overlays in base64 format.
"""

from fastapi import FastAPI, UploadFile, File
from infer import Inference
import numpy as np
import cv2
import base64

app = FastAPI()

infer = Inference(model_path="./saved_model/mobile_unet.keras")


@app.get("/")
async def root():
    """
    Health check / welcome endpoint.

    Returns:
        dict: A simple welcome message to confirm that the API is running.
    """
    return {"message": "Hello from mobile-unet!"}


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Run lane segmentation on an uploaded image.

    This endpoint takes an input image file, preprocesses it, performs inference
    using the loaded model, and returns a segmented image overlay encoded as a
    base64 string.

    Args:
        file (UploadFile): The uploaded image file to segment.

    Returns:
        dict: A dictionary with a single key:
            - "segmented_image" (str): Base64-encoded segmented overlay image.
    """
    # read image
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not decode image"}

    mask = infer.predict(image.copy())

    # overlay
    image = cv2.resize(image, (224, 224))
    red_mask = np.zeros((224, 224, 3), dtype=np.uint8)
    red_mask[:, :, 2] = mask[:, :, 0]
    seg_img = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0.0)

    # encode mask
    _, buffer = cv2.imencode(".png", seg_img)
    b64_seg = base64.b64encode(buffer).decode("utf-8")

    return {"segmented_base64": b64_seg}


if __name__ == "__main__":
    import uvicorn
    import tensorflow as tf

    # uvicorn.run(app, host='0.0.0.0', port=8000)
    print(tf.config.list_physical_devices("GPU"))
