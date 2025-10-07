import streamlit as st
import cv2
import numpy as np

import requests
import base64
import time

## ------------------------------------------
##            IMPORTANT
## Ensure docker container is running with:
## docker run -p 8000:8000 {image_name}
## ------------------------------------------

url = "http://0.0.0.0:8000/segment"

st.title("Lane Segmentation App")

# create a two columns
col1, col2 = st.columns(2)

uploaded_file = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"]
)  # file uploader
segment_button = st.button(
    "Segment Image", use_container_width=True
)  # button to segment image

# left frame for input image
with col1:
    st.header("Input Image")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        st.image(image, channels="BGR")
        seg_img = None

        # preprocess the image and predict
        if segment_button:
            if seg_img is not None:
                del seg_img
            orig_image = image.copy()
            start_time = time.time()
            response = requests.post(url, files={"file": uploaded_file.getvalue()})
            end_time = time.time()

            data = response.json()
            seg_b64 = data["segmented_base64"]

            # Decode back to OpenCV image
            seg_bytes = base64.b64decode(seg_b64)
            seg_np = np.frombuffer(seg_bytes, np.uint8)
            seg_img = cv2.imdecode(seg_np, cv2.IMREAD_COLOR)
            seg_img = cv2.resize(seg_img, (w, h), interpolation=cv2.INTER_CUBIC)

    else:
        st.write("No image uploaded yet.")

# right frame for segmented image
with col2:
    st.header("Segmented Image")

    if uploaded_file is not None and segment_button:
        st.image(seg_img, channels="BGR")
        st.write(f"Segmentation Time: {end_time - start_time:.3f} seconds")
