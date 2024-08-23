from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import lightning as L

from saws.config.train_config import TrainConfig
from saws.pretrain import main

from warms import (
    CANVAS_BASE_PATH,
    DATASET_MAP,
    ExpCanvas,
    prepare_data_handler_from_file
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--canvas_access",
        type=str,
        default="global",
        help="The key to decide the access point of the experiment configuration",
    )
    parser.add_argument(
        "--output_tree",
        type=str,
        default="test/run1",
        help="Creates a subdirectory tree starting from `results_root` in canvas configuration",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="wikitext",
        choices=["wikitext", "slimpajama"],
        help="Dataset name to load the data configuration",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default=None,
        help="Training configuration file."
    )
    # TODO: implement the feature below,
    # parser.add_argument(
    #     "--update_train_template",
    #     action="store_true",
    #     help=(
    #         "If True, updates the training template from the canvas configuration with the with "
    #         "the training configuration passed as `train_config_path`.\n"
    #         "If False, uses the `train_config_path` as the training template."
    #     )
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # if args.update_train_template:  # TODO: Implement this

    # Setting experiment canvas for path management
    canvas = ExpCanvas(CANVAS_BASE_PATH, args.canvas_access)

    # Loading the training configuration
    # if no arguments are passed, the default training template is used from the experiment canvas
    train_config = TrainConfig.from_path(
        args.train_config_path
        if args.train_config_path is not None
        else canvas.train_template
    )

    # Load the data configuration
    data_config = prepare_data_handler_from_file(
        data_config_path=canvas.data_handler_root / DATASET_MAP(args.dataset),
        train_config=train_config,
        root_data_path=canvas.data_root
    )

    pprint(data_config)
    pprint(train_config)

    # Interfacing `saws` (litgpt wrapper) for pretraining
    fabric = L.Fabric(devices="auto", strategy="auto")
    result_dict = main(
        fabric=fabric,
        data=data_config,
        train_args=train_config,
        out_dir=canvas.results_root / args.output_tree  # uses the canvas info as parent directory
    )
    print(f"{'=' * 20} Run done {'=' * 20}")
    pprint(result_dict)
# end of file