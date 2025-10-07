from . import config as cfg
from .utils import create_folder
import kagglehub
import os
from pathlib import Path


def setup_kaggle_credentials():
    """
    Sets up Kaggle credentials dynamically by pointing the KAGGLE_CONFIG_DIR
    environment variable to the parent directory of the provided kaggle.json file.
    """

    json_path = Path(cfg.KAGGLE_CREDENTIAL_JSON_PATH)
    if not json_path.is_file():
        raise FileNotFoundError(f"kaggle.json not found at: {json_path}")

    credential_dir = json_path.parent
    os.environ["KAGGLE_CONFIG_DIR"] = str(credential_dir)


def download_dataset():
    """
    Download the dataset specified in the configuration file from Kaggle.
    """

    # get the download directory and dataset name from the config file
    new_cache_path = cfg.NEW_CACHE_DIR
    dataset_name = cfg.DATASET_NAME

    # Set the environment variable for the current process
    os.environ["KAGGLEHUB_CACHE"] = new_cache_path

    # create directory at new file path
    path = create_folder(new_cache_path)

    # download the kaggle dataset
    try:

        path = kagglehub.dataset_download(dataset_name)
        cfg.set_dataset_path(path)
        print("Dataset Downloaded at :", path)

    except Exception as e:
        print(f"Error Occured: {str(e)}")
