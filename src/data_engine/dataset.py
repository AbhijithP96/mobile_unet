import tensorflow as tf
import pandas as pd
import os

from . import config as cfg


class TuSimple:
    """
    Utility class to create TensorFlow Datasets for training and validation
    from image and mask file paths
    """

    def __init__(
        self, paths_df: pd.DataFrame, batch_size: int = 16, augmenter: list = None
    ):
        """
        Initialie the utility class.

        Args:
            paths_df (pd.DataFrame): Dataframw with image and mask paths
            batch_size (int, optional): Batch size for training/validation. Defaults to 16.
        """

        self.paths_df = paths_df
        self.image_size = tf.convert_to_tensor(
            cfg.ENCODER_IMAGE_SIZE[:2], dtype=tf.int32
        )
        self.batch_size = tf.convert_to_tensor(batch_size, dtype=tf.int64)
        self.dataset_path = cfg.NEW_DATASET_PATH + "/TuSimple_Processed"

        self.augmenter = (
            augmenter if augmenter is not None else lambda img, mask: (img, mask)
        )

    def _parse_input(self, image_path, mask_path, split):
        """
        Load and preprocess a single image-mask pair.
        """

        # convert paths from relative to absolute
        prefix = tf.constant(self.dataset_path)
        image_path = tf.strings.join([prefix, split, image_path], separator="/")
        mask_path = tf.strings.join([prefix, split, mask_path], separator="/")

        # load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, dtype=tf.float32)

        # load mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.resize(mask, self.image_size, method="nearest")
        mask = tf.cast(mask, dtype=tf.float32)

        if split == "train" and self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        # normalize the data
        image = image / 127.5 - 1.0  # mobilenet requires data to be in [-1,1]
        mask = mask / 255.0  # decoder output is sigmoid, hence normalises to [0,1]

        return image, mask

    def get_dataset(self, split: str = "train") -> tf.data.Dataset:
        """
        Create a Tensorflow dataser for the specified split.

        Args:
            split (str, optional): Which split to return. Defaults to 'train'.

        Returns:
            tf.data.Dataset: TensorFlow dataset yielding (image, mask) pairs.
        """

        if split not in ["train", "val", "test"]:
            raise ValueError(f" Invalid split: {split}, must be either train or val")

        split_df = self.paths_df[self.paths_df["split"] == split]

        split_value = split_df["split"].values
        image_paths = split_df["image_path"].values
        mask_paths = split_df["mask_path"].values

        ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, split_value))
        ds = ds.map(self._parse_input, num_parallel_calls=tf.data.AUTOTUNE)

        # optimize data calls for faster training
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds
