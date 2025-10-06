# Kaggle credentials and download directory
KAGGLE_CREDENTIAL_JSON_PATH = "/home/basilisk/Downloads/kaggle.json"
NEW_CACHE_DIR = "/media/basilisk/75EEEA7B0189DFCA/dataset"

# Dataset Parameters
DATASET_NAME = "manideep1108/tusimple"
NEW_DATASET_PATH = NEW_CACHE_DIR + "/TuSimple/datasets/manideep1108/tusimple/versions/5"
TRAIN_DATA_SPLIT = 0.9

# image parameters
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

ENCODER_IMAGE_SIZE = (224, 224, 3)


def get_credentials_json_path():
    return KAGGLE_CREDENTIAL_JSON_PATH


def set_credentials_json_path(path):
    KAGGLE_CREDENTIAL_JSON_PATH = path


def get_dataset_name():
    return DATASET_NAME


def set_dataset_name(dataset):
    DATASET_NAME = dataset


def get_dataset_path():
    return NEW_DATASET_PATH


def set_dataset_path(path):
    NEW_DATASET_PATH = path


def get_cache_dir():
    return NEW_CACHE_DIR


def set_cache_dir(path):
    NEW_CACHE_DIR = path


def set_image_size(size):
    IMAGE_HEIGHT = size[0]
    IMAGE_WIDTH = size[1]


def get_image_size(size):
    return IMAGE_HEIGHT, IMAGE_WIDTH


def set_encoder_size(size):
    ENCODER_IMAGE_SIZE = tuple(size)


def get_encoder_size():
    return ENCODER_IMAGE_SIZE
