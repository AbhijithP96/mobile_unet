import src.data_engine as de
import argparse


def update_config(args):
    """
    To update the config file to be used for data loading and processing
    """

    if args.credentials:
        de.config.set_credentials_json_path(args.credentials)

    if args.cache:
        de.config.set_cache_dir(args.cache_dir)

    if args.dataset:
        de.config.set_dataset_name(args.dataset)

    if args.dest:
        de.config.set_dataset_path(args.dest)

    if args.imsz:
        de.config.set_image_size(args.imsz)

    if args.ensz:
        size = (args.ensz, args.ensz)
        de.config.set_image_size(size)


def kaggle_login_and_download():
    """
    An high level function to help user to download kaggle datasets.
    """

    de.kaggle_utils.setup_kaggle_credentials()
    de.kaggle_utils.download_dataset()


def main():

    parser = argparse.ArgumentParser(description="Data Management CLI")

    # --- Top-level flags ---
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update configuration file with provided arguments.",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Login to Kaggle and download dataset specified in config.",
    )

    # --- Config update options ---
    parser.add_argument(
        "--credentials", type=str, help="Path to Kaggle API credentials JSON file."
    )
    parser.add_argument("--cache_dir", type=str, help="Path to cache directory.")
    parser.add_argument(
        "--dataset", type=str, help="Name of the dataset (e.g. 'tusimple')."
    )
    parser.add_argument(
        "--dest", type=str, help="Destination path where dataset will be stored."
    )
    parser.add_argument(
        "--imsz",
        nargs=2,
        type=int,
        help="Image size as two integers, e.g. --imsz 224 224",
    )
    parser.add_argument("--ensz", type=int, help="Embedding size (int).")

    args = parser.parse_args()

    # --- Execute based on flags ---
    if args.update:
        update_config(args)

    if args.kaggle:
        kaggle_login_and_download()


if __name__ == "__main__":
    main()
