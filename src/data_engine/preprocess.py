import numpy as np
import pandas as pd
import cv2
import os
import shutil

from tqdm import tqdm
from typing import Tuple
from sklearn.model_selection import train_test_split

from . import config as cfg
from .utils import create_folder

DATASET_PATH = cfg.NEW_DATASET_PATH
TRAIN_SPLIT = cfg.TRAIN_DATA_SPLIT


def read_and_split_annotations(
    dataset: str = "train_set",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load lane annotation labels from all JSON files in the dataset and
    split them into training and validation sets.

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]:
            - **train_df**: DataFrame containing training samples.
            - **val_df**: DataFrame containing validation samples.
    """

    label_path = os.path.join(DATASET_PATH, "TUSimple", dataset)
    label_files = [
        os.path.join(label_path, file)
        for file in os.listdir(label_path)
        if file.endswith(".json")
    ]  # get all json file paths

    # read all json files
    dataframes = [pd.read_json(json_path, lines=True) for json_path in label_files]
    all_data = pd.concat(dataframes, ignore_index=True)

    if dataset[:-4] == "train":
        # split into train and validation sets for the train set
        train_data, val_data = train_test_split(
            all_data, test_size=1 - TRAIN_SPLIT, random_state=40, shuffle=True
        )

        return train_data, val_data

    else:
        # return the test data without splitting
        return all_data


def generate_lane_masks(row: pd.Series) -> Tuple[np.ndarray, str]:
    """
    Create a binary lane mask for a single image.

    Args:
        row (pd.Series): A single row from the dataset coantaining the annotation of a single image

    Returns:
        Tuple[np.ndarray,str]:
            - **mask**: Binary mask with lanes = 1 and background = 0
            - **raw_file**: Path to the image file realtive to the dataset path.
    """
    h, w = cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH

    # create a dummy image for the mask
    mask = np.zeros((h, w, 1), dtype=np.uint8)

    # extract the data from the row
    h_samples = row.h_samples
    lanes = row.lanes
    raw_file = row.raw_file

    # generate mask: lane =1 and background = 0
    for lane in lanes:
        # exclude -2 datapoints in lane
        h_sample_filtered = np.array([y for x, y in zip(lane, h_samples) if x != -2])
        lane_filtered = np.array([x for x in lane if x != -2])

        # create a array of 2D image points for the lane mask
        lane_points = np.c_[lane_filtered, h_sample_filtered]
        lane_points = lane_points.reshape((-1, 1, 2)).astype(
            np.int32
        )  # convert the points as expected by the cv2.polylines

        # update the dummy mask
        cv2.polylines(mask, [lane_points], isClosed=False, color=255, thickness=10)

    return mask, raw_file


def process_and_save(
    dataset: pd.DataFrame, split: str = "train", path_list: list = None
):
    """
    Process the dataset to generate lane masks and save train/val splits.

    Args:
        dataset (pd.DataFrame): Dataframe with image relative paths and annotations
        split (str, optional): Name of the dataset split to process. Accepted values are `"train"` or `"val"`. Defaults to 'train'.
    """

    # create folders at the new path
    new_path = os.path.join(DATASET_PATH, "TuSimple_Processed", split)
    new_path = create_folder(new_path)

    # create image and mask sub directories
    image_dir = create_folder(new_path + "/images")
    mask_dir = create_folder(new_path + "/masks")

    # create a dataframe to save the image path and mask path relative to new dataset path
    paths = path_list if path_list is not None else []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        # get the mask and relative image path
        mask, image_rel_path = generate_lane_masks(row)

        # get the actual image path
        split_name = "train" if split in ["train", "val"] else "test"
        raw_image_path = os.path.join(
            DATASET_PATH, "TUSimple", f"{split_name}_set", image_rel_path
        )

        # save the mask and image to the new dataset folder
        tmp = image_rel_path.split("/")
        filename = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"
        cv2.imwrite(os.path.join(mask_dir, filename), mask)
        shutil.copy(raw_image_path, os.path.join(image_dir, filename))

        # append to paths
        paths.append(
            {
                "split": split,
                "image_path": f"images/{filename}",
                "mask_path": f"masks/{filename}",
            }
        )

    return paths


def save_paths_df(paths, split):
    # save the paths
    paths_df = pd.DataFrame(paths, columns=["split", "image_path", "mask_path"])
    paths_df.to_csv(
        os.path.join(
            os.path.join(DATASET_PATH, "TuSimple_Processed"), f"paths_{split}.csv"
        )
    )


def preprocess(
    path: str = None, split: str = "train", finished: bool = False
) -> pd.DataFrame:
    """
    Run all the required prerocess steps using the downoaded dataset

    Returns:
        pd.DataFrame: Dataframe containing the paths of images and maks
    """

    if finished:
        df = (
            pd.read_csv(path)
            if path
            else pd.read_csv(
                os.path.join(
                    os.path.join(DATASET_PATH, "TuSimple_Processed"),
                    f"paths_{split}.csv",
                )
            )
        )
        return df

    data = read_and_split_annotations(split + "_set")
    if split == "train":
        paths = process_and_save(data[0], split="train")
        paths = process_and_save(data[1], split="val", path_list=paths)
        save_paths_df(paths, split=split)

    else:
        paths = process_and_save(data, split="test")
        save_paths_df(paths, split=split)

    return pd.DataFrame(paths, columns=["split", "image_path", "mask_path"])


if __name__ == "__main__":

    train_data, val_data = read_and_split_annotations()
    paths = process_and_save(train_data, split="train")
    paths = process_and_save(val_data, split="val", path_list=paths)

    save_paths_df(paths)
