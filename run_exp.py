"""
Experiment Runner for Lane Segmentation Training

This module automates running multiple training experiments for the
lane segmentation model using configurations defined in a JSON file.
Each experiment is executed as a separate process to ensure isolated
resource usage and prevent memory buildup between runs.
"""

import argparse
import json
from train import train_unet
import utils
import time
import multiprocessing as mp


def get_args():
    """
    Parse command-line arguments for experiment execution.

    Returns:
        argparse.Namespace: Parsed arguments with the following attribute:
            - json (str): Path to the JSON file containing experiment definitions.
              Defaults to "./exp.json".
    """

    parser = argparse.ArgumentParser(
        description="Program to run different experiments in a json"
    )
    parser.add_argument(
        "--json", type=str, default="./exp.json", help="Path to the json file"
    )

    args = parser.parse_args()

    return args


def train_exp(args):
    """
    Execute a series of training experiments defined in a JSON configuration file.

    Reads experiment definitions (name, description, MLflow experiment name)
    and corresponding runs (hyperparameter sets) from the provided JSON file.
    Each run is launched as a separate process to avoid memory accumulation
    and ensure reproducibility.
    """

    with open(args.json, "r") as f:
        exp_config = json.load(f)

    experiments = exp_config["experiments"]

    results = {}

    for exp in experiments:
        name = exp["name"]
        desc = exp["description"]
        exp_name = exp["mlflow_experiment"]

        print(f"\n{'='*60}")
        print(f"Starting Experiment: {name}")
        print(f"Description: {desc}")
        print(f"\n{'='*60}")

        runs = exp["runs"]

        for run in runs:

            print("Starting New Run")

            config = {"mlflow": {"experiment_name": exp_name, "description": desc}}
            config.update(run)

            p = mp.Process(target=train_unet, args=(config,))
            p.start()
            p.join()

            time.sleep(20)

    return results


if __name__ == "__main__":
    args = get_args()
    res = train_exp(args)

    utils.dump_json(res, name="exp_results")
