import pytest
import numpy as np
import tensorflow as tf

import utils
from utils import dice_coef
from src.losses import BceDiceLoss, FocalDiceLoss


## ---------------------------------------------------
#               DICE LOSS
## ---------------------------------------------------


def test_dice_coefficient_perfect_match():
    """
    Test Dice coefficient with perfect prediciton match.
    """
    # Create identical ground truth and prediciton
    y_true = tf.ones((1, 224, 224, 1), dtype=tf.float32)
    y_pred = tf.ones((1, 224, 224, 1), dtype=tf.float32)

    # calcualte dice coefficient using the implement function
    dice_score = dice_coef(y_true, y_pred)

    # Assert perfect match
    assert (
        abs(dice_score.numpy() - 1.0) < 1e-6
    ), f"Perfect match should yield dice=1.0, got {dice_score.numpy()}"


def test_dice_coefficient_no_overlap():
    """
    Test Dice coefficient with no overlap between prediction and ground truth.
    """
    # Create identical ground truth and prediciton
    y_true = tf.zeros((1, 224, 224, 1), dtype=tf.float32)
    y_pred = tf.ones((1, 224, 224, 1), dtype=tf.float32)

    # calcualte dice coefficient using the implement function
    dice_score = dice_coef(y_true, y_pred)

    # Assert perfect match
    assert (
        abs(dice_score.numpy() - 0.0) < 1e-6
    ), f"No overlap should yield dice=0.0, got {dice_score.numpy()}"


def test_dice_loss_perfect_match():
    """
    Test Dice Loss with perfect prediciton match.
    """
    # Create identical ground truth and prediciton
    y_true = tf.ones((1, 224, 224, 1), dtype=tf.float32)
    y_pred = tf.ones((1, 224, 224, 1), dtype=tf.float32)

    # get dice loss
    dice_loss = utils.get_loss_fn(loss="dice")

    # calcualte dice coefficient using the implement function
    dice_loss = dice_loss(y_true, y_pred)

    # Assert perfect match
    assert (
        abs(dice_loss.numpy() - 0.0) < 1e-6
    ), f"Perfect match should yield dice loss=0.0, got {dice_loss.numpy()}"


def test_dice_loss_no_overlap():
    """
    Test Dice Loss with no overlap between prediction and ground truth.
    """
    # Create identical ground truth and prediciton
    y_true = tf.zeros((1, 224, 224, 1), dtype=tf.float32)
    y_pred = tf.ones((1, 224, 224, 1), dtype=tf.float32)

    # get dice loss
    dice_loss = utils.get_loss_fn(loss="dice")

    # calcualte dice coefficient using the implement function
    dice_score = dice_loss(y_true, y_pred)

    # Assert perfect match
    assert (
        abs(dice_score.numpy() - 1.0) < 1e-6
    ), f"No Overlap should yield dice=1.0, got {dice_score.numpy()}"


def test_dice_loss_coef_range():
    """
    Test range of Dice coefficient and loss values
    """

    y_true = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)
    y_pred = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)

    # get the dice loss
    dice_loss = utils.get_loss_fn(loss="dice")

    # calculate the loss and coefficient
    coef_value = dice_coef(y_true, y_pred)
    loss_value = dice_loss(y_true, y_pred)

    # Assert the range
    assert 0.0 <= coef_value <= 1.0, "Dice coefficient should be within [0,1]"
    assert 0.0 <= loss_value <= 1.0, "Dice loss should be within [0, 1]"


def test_dice_loss_gradients():
    """
    Test that Dice loss produces valid gradients for backpropagation.
    """
    y_true = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)

    # get the dice loss
    dice_loss = utils.get_loss_fn(loss="dice")

    # Use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        y_pred = tf.Variable(
            tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)
        )
        tape.watch(y_pred)
        loss_value = dice_loss(y_true, y_pred)

    gradients = tape.gradient(loss_value, y_pred)

    # Check gradients are valid
    assert gradients is not None, "Gradients should not be None"
    assert tf.reduce_all(tf.math.is_finite(gradients)), "Gradients should be finite"
    assert not tf.reduce_all(tf.equal(gradients, 0)), "Gradients should not all be zero"


## ---------------------------------------------------
#               FOCAL-DICE LOSS
## ---------------------------------------------------


def test_focal_dice_combination():
    """
    Test combined Focal-Dice loss.
    """
    y_true = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)
    y_pred = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)

    # get the dice loss
    dice_loss = utils.get_loss_fn(loss="dice")
    focal_loss = utils.get_loss_fn(loss="focal")

    # Calculate individual losses
    dice_loss_val = dice_loss(y_true, y_pred)
    focal_loss_val = focal_loss(y_true, y_pred)

    # test different weight combinations
    weights = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for focal_wt, dice_wt in weights:

        combined_loss = FocalDiceLoss(focal_weight=focal_wt, dice_weight=dice_wt)(
            y_true, y_pred
        )

        expected_loss = focal_wt * focal_loss_val + dice_wt * dice_loss_val

        assert (
            abs(combined_loss.numpy() - expected_loss.numpy()) < 1e-6
        ), f"Combined loss should match weighted sum: {combined_loss.numpy()} vs {expected_loss.numpy()}"


def test_combined_focal_dice_loss_gradients():
    """
    Test gradient computation for combined loss.

    """
    shape = (4, 224, 224, 1)
    y_true = tf.random.uniform(shape, 0, 1, dtype=tf.float32)

    combined_loss = FocalDiceLoss(focal_weight=0.3, dice_weight=0.7)

    with tf.GradientTape() as tape:
        y_pred = tf.Variable(tf.random.uniform(shape, 0, 1, dtype=tf.float32))
        tape.watch(y_pred)
        loss_val = combined_loss(y_true, y_pred)

    gradients = tape.gradient(loss_val, y_pred)

    assert gradients is not None, "Gradients should not be None"
    assert tf.reduce_all(tf.math.is_finite(gradients)), "Gradients should be finite"


## ---------------------------------------------------
#               BCE-DICE LOSS
## ---------------------------------------------------


def test_focal_dice_combination():
    """
    Test combined BCE-Dice loss.
    """
    y_true = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)
    y_pred = tf.random.uniform((4, 224, 224, 1), 0, 1, dtype=tf.float32)

    # get the dice loss
    dice_loss = utils.get_loss_fn(loss="dice")
    bce_loss = utils.get_loss_fn(loss="bce")

    # Calculate individual losses
    dice_loss_val = dice_loss(y_true, y_pred)
    bce_loss_val = bce_loss(y_true, y_pred)

    # test different weight combinations
    weights = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for bce_wt, dice_wt in weights:

        combined_loss = BceDiceLoss(bce_weight=bce_wt, dice_weight=dice_wt)(
            y_true, y_pred
        )

        expected_loss = bce_wt * bce_loss_val + dice_wt * dice_loss_val

        assert (
            abs(combined_loss.numpy() - expected_loss.numpy()) < 1e-6
        ), f"Combined loss should match weighted sum: {combined_loss.numpy()} vs {expected_loss.numpy()}"


def test_combined_bce_dice_loss_gradients():
    """
    Test gradient computation for combined loss.

    """
    shape = (4, 224, 224, 1)
    y_true = tf.random.uniform(shape, 0, 1, dtype=tf.float32)

    combined_loss = BceDiceLoss(bce_weight=0.3, dice_weight=0.7)

    with tf.GradientTape() as tape:
        y_pred = tf.Variable(tf.random.uniform(shape, 0, 1, dtype=tf.float32))
        tape.watch(y_pred)
        loss_val = combined_loss(y_true, y_pred)

    gradients = tape.gradient(loss_val, y_pred)

    assert gradients is not None, "Gradients should not be None"
    assert tf.reduce_all(tf.math.is_finite(gradients)), "Gradients should be finite"
