"""This script is designed to update the model config (for scale) given a train config.
"""

import argparse
from copy import deepcopy
import lightning as L
from litgpt.config import Config
from pathlib import Path

from saws import TrainConfig, main

from warms import (
    CANVAS_BASE_PATH,
    DATASET_MAP,
    ExpCanvas,
    prepare_data_handler_from_file
)


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
        help="The optimal LR at the base scale"
    )   
    parser.add_argument(
        "--target_scale",
        type=str,
        required=True,
        help="The path to target scale model config"
    )

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
    train_config = TrainConfig.from_path(
        canvas.train_template
        if args.train_template is None
        else Path(args.train_template)
    )
    _train_config = deepcopy(train_config)  # for debugging purposes
    data_config = prepare_data_handler_from_file(
        data_config_path=canvas.data_handler_root / DATASET_MAP(args.dataset),
        train_config=train_config,
        root_data_path=canvas.data_root
    )
    model_config = Config.from_file(Path(args.target_scale))

    # Updating the model config in train config
    for k, v in train_config.model_config.to_dict().items():
        if k in model_config.__dict__:
            setattr(train_config.model_config, k, getattr(model_config, k, v))
    train_config.model_config.d_model = model_config.n_embd
    train_config.block_size = model_config.block_size
    if args.base_lr is not None and args.mup_base is not None:
        train_config.mup_base_shape_path = Path(args.mup_base)
        train_config.max_lr = args.base_lr  # crucial for muP to work properly
    
    # Running
    fabric = L.Fabric(devices="auto", strategy="auto")
    result_dict = main(
        fabric=fabric,
        data=data_config,
        train_args=train_config,
        out_dir=canvas.results_root / args.output_tree  # uses the canvas info as parent directory
    )
# end of file