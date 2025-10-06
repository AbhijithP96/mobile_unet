"""
Custom loss functions for lane segmentation tasks.

This module implements specialized loss functions optimized for binary
segmentation tasks, specifically designed for
lane detection.

Key Features:
    - Combined Loss: Weighted combination of dice loss with either binary cross entropy loss or
    focal loss for optimal performance

Classes:
    BceDiceLoss
    FocalDiceLoss

"""

import keras
import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable()
class BceDiceLoss(keras.losses.Loss):
    """
    Combines Dice Loss with Binary Cross-Entropy (BCE) Loss.

    Example Usage:
        loss_fn = BceDiceLoss(bce_weight=0.5, dice_weight=0.5)
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, **kwargs):

        super().__init__(name="BinaryCrossEntropy_and_Dice", **kwargs)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss_fn = keras.losses.BinaryCrossentropy()
        self.dice_loss_fn = keras.losses.Dice()

    def call(self, y_true, y_pred):

        bce_loss = self.bce_loss_fn(y_true, y_pred)  # binary cross entropy loss
        dice_loss = self.dice_loss_fn(y_true, y_pred)  # dice loss

        combined_loss = (
            self.bce_weight * bce_loss + self.dice_weight * dice_loss
        )  # weighted combined loss

        return combined_loss

    def get_config(self):
        config = super().get_config()
        config.pop("name", None)
        config.update(
            {
                "bce_weight": self.bce_weight,
                "dice_weight": self.dice_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class FocalDiceLoss(keras.losses.Loss):
    """
    Combines Dice Loss with Focal Loss.

    Example Usage:
        loss_fn = FocalDiceLoss(focal_weight=0.6, dice_weight=0.4)
        loss = loss_fn(y_true, y_pred)
    """

    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5, **kwargs):

        super().__init__(name="BinaryCrossEntropy_and_Dice", **kwargs)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal_loss_fn = keras.losses.BinaryFocalCrossentropy()
        self.dice_loss_fn = keras.losses.Dice()

    def call(self, y_true, y_pred):

        focal_loss = self.focal_loss_fn(y_true, y_pred)  # focal loss calculation
        dice_loss = self.dice_loss_fn(y_true, y_pred)  # dice loss calculation

        combined_loss = (
            self.focal_weight * focal_loss + self.dice_weight * dice_loss
        )  # weighted combined loss

        return combined_loss

    def get_config(self):
        config = super().get_config()
        config.pop("name", None)
        config.update(
            {
                "focal_weight": self.focal_weight,
                "dice_weight": self.dice_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
