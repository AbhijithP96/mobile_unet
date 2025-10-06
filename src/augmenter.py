"""
Augmentation Module for Lane Segmentation

This module provides utilities for creating and applying data augmentation
pipelines to images and masks in segmentation tasks using albumentations package.
"""

import tensorflow as tf
import numpy as np
import albumentations as A


def create_augmenter() -> A.Compose:
    """
    Create and configure an augmentation pipeline.

    Returns:
        augmenter: A callable augmentation pipeline that can be applied
        to images and masks.
    """

    augmenter = A.Compose(
        [
            # GEOMETRIC
            # horizontal flip
            A.HorizontalFlip(),
            # rotation, scaling and translation
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                shear=(-5, 5),
            ),
            # perspective transformation
            A.Perspective(scale=(0.05, 0.15), keep_size=True),
            # PHOTOMETRIC
            # brightness and contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            # non -linear changer,
            A.RandomGamma(p=0.2),
            # noise
            A.GaussNoise(p=0.3),
            # Motion or Gaussian Blur
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                ],
                p=0.2,
            ),
            # ENVIRONMENTAL
            # weather change
            A.OneOf(
                [
                    A.RandomRain(p=1.0),
                    A.RandomFog(p=1.0),
                ],
                p=0.1,
            ),
        ]
    )

    return augmenter


def apply_augmentation(image, mask, augmenter=None):
    """
    Apply the given augmentation pipeline to an image and its mask.
    """

    if augmenter is None:
        augmenter = create_augmenter()

    def tf_augment(input, output):

        # Convert to uint8 (imgaug's preferred format)
        image_np = np.clip(input, 0, 255).astype(np.uint8)
        mask_np = np.clip(output, 0, 255).astype(np.uint8)

        # apply augmentation
        transformed = augmenter(image=image_np, mask=mask_np)
        image_aug = transformed["image"]
        mask_aug = transformed["mask"]

        # Convert back to float32 and maintain [0, 255] range
        image_aug = image_aug.astype(np.float32)
        mask_aug = mask_aug.astype(np.float32)

        return image_aug, mask_aug

    aug_image, aug_mask = tf.py_function(
        func=tf_augment, inp=[image, mask], Tout=[tf.float32, tf.float32]
    )

    # Set shapes for TensorFlow graph compatibility
    aug_image.set_shape(image.shape)
    aug_mask.set_shape(mask.shape)

    return aug_image, aug_mask


if __name__ == "__main__":
    image = tf.io.read_file(
        "/home/basilisk/Documents/portfolio/mobilenet_unet/data/0313-1_120_20.jpg"
    )
    image = tf.image.decode_jpeg(image, channels=3)

    mask = tf.io.read_file(
        "/home/basilisk/Documents/portfolio/mobilenet_unet/data/0313-1_120_20 _mask.jpg"
    )
    mask = tf.image.decode_jpeg(mask, channels=1)

    res = apply_augmentation(image, mask)

    print(res[1].shape)
