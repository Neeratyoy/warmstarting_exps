"""This script is designed to update the model config (for scale) given a train config.
"""

import argparse
import yaml
from copy import deepcopy
import lightning as L
from litgpt.config import Config
from pathlib import Path

from saws.config.yaml_utils import path_constructor
from saws import TrainConfig, main

from warms import (
    CANVAS_BASE_PATH,
    DATASET_MAP,
    ExpCanvas,
    prepare_data_handler_from_file
)


def warmstart_parser(args: argparse.Namespace, train_config: TrainConfig) -> TrainConfig:
    if args.warmstart:
        train_config["warmstart_config"]["activate"] = True
        train_config["warmstart_config"]["warmstart_type"] = args.warmstart_type
        train_config["warmstart_config"]["buffer_logging"] = args.warmstart_log_buffer
        if args.warmstart_base_path is not None:
            train_config["warmstart_config"]["base_model_path"] = args.warmstart_base_path
        else:
            assert train_config["warmstart_config"]["base_model_path"] is not None, "Base model path is required for warmstarting."
    return train_config


def get_args():
    parser = argparse.ArgumentParser(description="Parser for generating MuP base files")

    parser.add_argument(
        "--canvas_access",
        type=str,
        default="global",
        help="The key to decide the access point of the experiment configuration",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset choice",
        choices=["wikitext", "slimpajama"]
    )
    parser.add_argument(
        "--output_tree",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--train_template",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mup_base",
        type=str,
        help="The path to the .bsh file for base muP scale"
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=None,
        help="The optimal LR at the base scale"
    )
    parser.add_argument(
        "--target_scale",
        type=str,
        default=None,
        help="The path to target scale model config"
    )

    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--warmstart_type", type=str, default="zeros")
    parser.add_argument("--warmstart_log_buffer", action="store_true")
    parser.add_argument("--warmstart_base_path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if hasattr(args, "base_lr"):
        assert hasattr(args, "mup_base"), "MuP base file path is required for when using base LR."
    if hasattr(args, "mup_base"):
        assert hasattr(args, "base_lr"), "Base LR is required for when using MuP base file."

    # Setting experiment canvas for path management
    canvas = ExpCanvas(CANVAS_BASE_PATH, args.canvas_access)

    # Loading
    yaml.SafeLoader.add_constructor('!path', path_constructor)
    with (canvas.train_template if args.train_template is None else Path(args.train_template)).open(
            encoding="utf-8") as yaml_file:
        train_config = yaml.safe_load(yaml_file)

    if args.target_scale is not None:
        with (Path(args.target_scale)).open(encoding="utf-8") as yaml_file:
            model_config = yaml.safe_load(yaml_file)
        if "max_micro_batch_size" in model_config:
            _max_micro_batch_size = model_config.pop("max_micro_batch_size")
            if _max_micro_batch_size is not None:
                train_config["max_micro_batch_size"] = _max_micro_batch_size

        train_config["model_config"] = model_config
        train_config["block_size"] = model_config["block_size"]

    # adjusting for muP
    if args.base_lr is not None:
        train_config["max_lr"] = args.base_lr  # crucial for muP to work properly
    if args.mup_base is not None:
        train_config["mup_base_shape_path"] = Path(args.mup_base)

    # adjusting for warmstarting
    train_config = warmstart_parser(args, train_config)

    train_config = TrainConfig(**train_config)

    data_config = prepare_data_handler_from_file(
        data_config_path=canvas.data_handler_root / DATASET_MAP(args.dataset),
        train_config=train_config,
        root_data_path=canvas.data_root
    )

    # Running
    fabric = L.Fabric(devices="auto", strategy="auto")
    result_dict = main(
        fabric=fabric,
        data=data_config,
        train_args=train_config,
        out_dir=canvas.results_root / args.output_tree  # uses the canvas info as parent directory
    )
# end of file
