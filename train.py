"""
Training Script for Lane Segmentation Model

This module orchestrates the end-to-end training process for the lane
segmentation model.
"""

from src.data_engine import preprocess, dataset
from src.nn.unet import UNet
from train_config import get_train_config
import tensorflow as tf
import mlflow
import os
import keras

from keras.callbacks import TensorBoard
from src.augmenter import apply_augmentation

import utils


def train_unet(json_dict: dict = {}) -> UNet:
    """
    Train a UNet model with hyperparameters and augmentations defined in config dict.
    Logs to MLflow if `log_to_mlflow` is True.
    """

    # get train configurations
    cfg_dict, cfg = get_train_config(json_dict)

    # get config artifact
    cfg_dict_path = utils.get_config_json(cfg_dict)

    # prepare the downloaded dataset
    paths_df = preprocess.preprocess(cfg.dataset.path, finished=True)

    # create the train and val dataset
    augmenter = apply_augmentation if cfg.dataset.augment else None
    data = dataset.TuSimple(
        paths_df, batch_size=cfg.hyperparams.batches, augmenter=augmenter
    )
    train_data = data.get_dataset(split="train")
    val_data = data.get_dataset(split="val")

    # set the parameters for the model
    input_size = cfg.hyperparams.input_size
    filter_list = cfg.hyperparams.filters
    epochs = cfg.hyperparams.epochs

    # build the model
    model = UNet(input_size, filter_list, mobile=cfg.decoder.mobile)

    model.build()
    model_summary_path = model.get_summary_path()

    # define the loss and optimizer
    # dice_loss = keras.losses.Dice()
    loss = utils.get_loss_fn(loss=cfg.loss.name, weights=cfg.loss.weights)
    # optimizer = keras.optimizers.Adam(learning_rate = 1e-3)
    scheduler = utils.build_learning_rate_schedule(
        required=cfg.optimizer.scheduler.required,
        schedule=cfg.optimizer.scheduler.name,
        base_lr=cfg.optimizer.lr,
        total_epochs=epochs,
    )

    optimizer = utils.build_optimizer(cfg.optimizer.name, scheduler)

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            utils.dice_coef,
        ],
    )

    if cfg.mlflow.log:
        # setup mlflow tracking
        track_uri = cfg.mlflow.tracking_uri
        mlflow.set_tracking_uri(track_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        desc = {"mlflow.note.content": cfg.mlflow.description}
        mlflow.set_experiment_tags(desc)

        run_desc = utils.dict_to_description(json_dict)

        # train the model with mlflow logging
        with mlflow.start_run(description=run_desc) as run:

            try:
                mlflow.log_artifact(model_summary_path, "model_summary")
                mlflow.log_artifact(cfg_dict_path, "training_configuration")
            finally:
                os.unlink(cfg_dict_path)
                os.unlink(model_summary_path)

            callbacks, best_model_path, temp_dir = utils.mlflow_callbacks()

            history = model.fit(
                train_data,
                epochs=epochs,
                validation_data=val_data,
                callbacks=[callbacks, mlflow.keras.MLflowCallback()],
            )

            utils.log_best_model_mlflow(
                history, best_model_path, temp_dir, epochs, cfg.mlflow.model_name
            )

            # run_id = run.info.run_id
            # run_name = run.info.run_name

        # clear cache
        del model, history
        keras.backend.clear_session()
        print(f"\n{'='*60}")
        print("Run Finished!!")
        print(f"\n{'='*60}")

    else:

        # run training using tensorboard
        # logging directory for the run
        log_dir = f"logs/{cfg.mlflow.experiment_name}_{cfg.mlflow.model_name}"
        os.makedirs(log_dir, exist_ok=True)

        board_callbacks = [
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,  # Disable to save space
                update_freq="epoch",
            )
        ]

        print(f"Training without MLFlow. TensorBoard logs: {log_dir}")
        print("No automoatic model saving!!")

        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=board_callbacks,
        )

        print(history)


if __name__ == "__main__":
    # running the training with hyperparameters obtained from experiments

    best_params_from_experiment = {
        "hyperparams": {"filters": [128, 64, 32, 16], "epochs": 300},
        "loss": {"name": "focal_dice", "weights": [0.3, 0.7]},
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "scheduler": {"required": True, "name": "cosine"},
        },
        "dataset": {"augment": True},
        "mlflow": {"experiment_name": "best_params", "model_name": "mobile_unet"},
    }

    train_unet(best_params_from_experiment)
