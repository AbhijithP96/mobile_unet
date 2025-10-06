"""

Utility Functions for Training and Experiment Management.

"""

import tensorflow as tf
from keras.utils import register_keras_serializable
from train_config import Config
from keras.models import load_model
import keras
import os
from src.losses import BceDiceLoss, FocalDiceLoss
import tempfile
from keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import numpy as np
import json


@register_keras_serializable(name="dice_coef")
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Coefficient for segmentation evaluation.

    The Dice Coefficient measures overlap between predicted and ground truth masks,
    ranging from 0 (no overlap) to 1 (perfect overlap).
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def build_optimizer(name: str, lr_schedule_or_value):
    """
    Create and return an optimizer based on the provided name and learning rate/schedule.
    """
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr_schedule_or_value)
    elif name == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr_schedule_or_value)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr_schedule_or_value)
    elif name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr_schedule_or_value, momentum=0.9)
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Must be one of ['adam', 'adamw', 'rmsprop', 'sgd']."
        )


def build_learning_rate_schedule(
    required: bool = False,
    base_lr: float = 1e-3,
    steps_per_epoch: int = 408,  # 3263 images / 8 batch size
    **kwargs,
):
    """
    Build a learning rate schedule if required, otherwise return a fixed learning rate.

    Args:
        required: Whether to use a schedule or fixed LR
        base_lr: Initial learning rate
        steps_per_epoch: Steps per epoch (default 408 for your dataset)
        **kwargs: Schedule-specific parameters

    Supported schedules:
        - 'exponential': Exponential decay
        - 'piecewise': Step-wise decay at specific epochs
        - 'cosine': Smooth cosine decay
        - 'onecycle': One-cycle learning rate policy
    """
    if not required:
        return base_lr

    schedule_name = kwargs.get("schedule", None)
    if schedule_name is None:
        raise ValueError(
            "Pass schedule='exponential'|'piecewise'|'cosine'|'cosine_restart'|"
            "'polynomial'|'onecycle' when required=True"
        )

    schedule_name = schedule_name.lower()

    # Convert epoch-based parameters to step-based for your dataset
    total_epochs = kwargs.get("total_epochs", 50)
    total_steps = steps_per_epoch * total_epochs

    if schedule_name == "exponential":
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr,
            decay_steps=kwargs.get("decay_steps", 1000),
            decay_rate=kwargs.get("decay_rate", 0.96),
            staircase=kwargs.get("staircase", False),
        )

    elif schedule_name == "piecewise":
        return keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=kwargs.get("boundaries", [1000, 2000]),
            values=kwargs.get("values", [base_lr, base_lr * 0.1, base_lr * 0.01]),
        )

    elif schedule_name == "cosine":
        # Cosine decay over full training
        decay_steps = kwargs.get("decay_steps", total_steps)

        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=decay_steps,
            alpha=kwargs.get("alpha", 0.01),  # Final LR = 1% of initial
        )

    else:
        raise ValueError(
            f"Unknown learning rate schedule '{schedule_name}'. Supported: ['exponential', 'piecewise', 'cosine']"
        )


def get_loss_fn(loss: str = "dice", **kwargs):
    """
    Retrieve a loss function by name.

    Args:
        loss name (str): Name of the loss function. Supported values:
            - "dice": Dice Loss
            - "bce": Binary Cross-Entropy Loss
            - "focal": Focal Loss
            - "bce_dice": Binary Cross-Entropy + Dice loss.
            - "focal_dice": Focal loss + Dice loss.
        **kwargs: Only supports `weight` (float) — a weighting factor to
            control the relative importance of combined loss function of Dice loss vs. BCE/Focal loss.
            Example: weight=0.7 gives 70% weight to Dice and 30% to BCE/Focal.

    Returns:
        Callable: A loss function that can be passed to `model.compile()`.

    Raises:
        ValueError: If the given loss function name is not supported.
    """

    loss = loss.lower()
    if loss == "dice":
        return keras.losses.Dice()

    elif loss == "bce":
        return keras.losses.BinaryCrossentropy()

    elif loss == "focal":
        return keras.losses.BinaryFocalCrossentropy()

    elif loss == "bce_dice":
        weights = kwargs.get("weights", [0.5, 0.5])
        return BceDiceLoss(bce_weight=weights[0], dice_weight=weights[1])

    elif loss == "focal_dice":
        weights = kwargs.get("weights", [0.5, 0.5])
        return FocalDiceLoss(focal_weight=weights[0], dice_weight=weights[1])
    else:
        raise ValueError("Loss Function cannot be determined")


def mlflow_callbacks(monitor="val_dice_coef", patience=25, min_delta=0.001):
    """
    Create a list of MLflow-compatible callbacks for model training.
    """
    # create a temporary directory for best model during training
    temp_dir = tempfile.mkdtemp()
    best_model_path = os.path.join(temp_dir, "best_model.keras")

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
            min_delta=min_delta,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
            verbose=1,
        ),
    ]

    return callbacks, best_model_path, temp_dir


def log_best_model_mlflow(
    history,
    best_model_path,
    temp_dir,
    epochs,
    model_name,
    monitor_metric="val_dice_coef",
):
    """
    Log the best-performing model to MLflow.

    Saves the model as an artifact in the current MLflow run, making it easy
    to retrieve the model.
    """

    # Calculate training statistics
    best_val_score = max(history.history[monitor_metric])
    best_epoch = history.history[monitor_metric].index(best_val_score) + 1
    final_epoch = len(history.history["loss"])

    # Log additional metrics to MLflow
    mlflow.log_metrics(
        {
            f"best_{monitor_metric}": best_val_score,
            "best_epoch": best_epoch,
            "total_epochs_trained": final_epoch,
            "early_stopped": final_epoch < epochs,
            "training_efficiency": best_epoch
            / final_epoch,  # How early the best model was found
        }
    )

    # Log the BEST model to MLflow
    if os.path.exists(best_model_path):
        print(f"Logging best model (epoch {best_epoch}) to MLflow...")

        # Load the best model
        best_model = load_model(best_model_path)

        # Log best model with metadata
        mlflow.tensorflow.log_model(
            model=best_model,
            input_example=np.random.rand(1, 224, 224, 3).astype(np.float32),
            registered_model_name=model_name,
            metadata={
                f"best_{monitor_metric}": float(best_val_score),
                "best_epoch": int(best_epoch),
                "total_epochs": int(final_epoch),
                "early_stopped": bool(final_epoch < epochs),
                "model_architecture": "MobileNet-UNet",
                "training_stopped_reason": (
                    "early_stopping" if final_epoch < epochs else "completed_all_epochs"
                ),
            },
        )

        print(f"Best model logged to MLflow ({monitor_metric}: {best_val_score:.4f})")
    else:
        print("Best model file not found, skipping model logging...")

    # Clean up temporary files
    try:
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        os.rmdir(temp_dir)
    except:
        pass


def dict_to_description(dict_to_convert):

    if not dict_to_convert:
        return "default_config"

    dict_to_convert = {k: v for k, v in dict_to_convert.items() if k != "mlflow"}

    return json.dumps(dict_to_convert, separators=(",", ":"))


def dump_json(dump_dict, name="dump"):

    with open(f"./{name}.json", "w") as f:
        json.dump(dump_dict, f, indent=4)

    print(f"JSON Saved with filename: {name}.json")


def get_config_json(cfg_dict):

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg_dict, tmp, indent=4)
        tmp_path = tmp.name

    return tmp_path
