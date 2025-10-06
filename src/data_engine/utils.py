import os
import numpy as np
import cv2


# utility to create a folders
def create_folder(path):

    os.makedirs(path, exist_ok=True)
    print(f"Using Folder at {path}")

    return path
