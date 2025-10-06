from src.data_engine import config
import copy

defaults = {
    "decoder": {"mobile": True},
    "hyperparams": {
        "input_size": config.ENCODER_IMAGE_SIZE,
        "filters": [512, 256, 128, 64],
        "num_classes": 1,
        "batches": 8,
        "epochs": 50,
    },
    "loss": {"name": "dice", "weights": None},
    "optimizer": {
        "name": "adam",
        "lr": 1e-3,
        "scheduler": {"required": False, "name": "piecewise"},
    },
    "dataset": {"path": None, "augment": True},
    "mlflow": {
        "log": True,
        "tracking_uri": "http://localhost:8080",
        "experiment_name": "default",
        "model_name": "mobile_unet_dev",
        "description": "default_experiment",
    },
}


class Config:
    def __init__(self, entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)

    def __getattr__(self, name):
        raise AttributeError


def get_train_config(json_dict):
    merged = _deep_merge(defaults, json_dict)
    return merged, Config(merged)


def _deep_merge(defaults: dict, updates: dict) -> dict:

    merged = copy.deepcopy(defaults)
    for k, v in updates.items():
        if k not in defaults:
            raise ValueError(f"Invalid key '{k}' in request JSON")

        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)

        else:
            merged[k] = v

    return merged
