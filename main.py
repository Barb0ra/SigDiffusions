import os
import argparse
import numpy as np
import yaml

from compute_signatures import compute_signatures
from train import train
from sample import sample
from invert_signatures import invert_signatures


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        type=str,
        choices=["run-all", "compute-sigs", "train", "sample", "invert-sigs"],
        help="action to perform",
    )
    parser.add_argument("name", type=str, default=None)

    parser.add_argument(
        "config_file", type=str, default=None, help="path of config file"
    )

    args = parser.parse_args()
    return args


def setup_logging_folders(logging_folders):
    for folder in logging_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def main():
    args = parse_args()

    config = load_yaml_config(args.config_file)

    setup_logging_folders(config["logging_folders"].values())

    np.random.seed(config["seed"])

    if args.action == "run-all":
        logsigs = compute_signatures(config, args.name)
        model, data_mean, data_std = train(config, args.name, logsigs)
        samples = sample(
            config, args.name, model, data_mean, data_std, logsigs.shape[1:]
        )
        invert_signatures(config, args.name, samples)

    elif args.action == "compute-sigs":
        compute_signatures(config, args.name)
    elif args.action == "train":
        train(config, args.name)
    elif args.action == "sample":
        sample(config, args.name)
    elif args.action == "invert-sigs":
        invert_signatures(config, args.name)


if __name__ == "__main__":
    main()
