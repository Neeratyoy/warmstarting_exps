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

from warms.utils.support import warmstart_parser


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
        default="slimpajama",
        help="Dataset choice",
        choices=["wikitext", "slimpajama", "slimpajama-small"]
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
    parser.add_argument("--warmstart_base_path", type=str, default=None)
    parser.add_argument("--base_model_step", type=int, default=None)
    parser.add_argument("--shrinking_factor", type=float, default=None)
    parser.add_argument("--perturbation_sigma", type=float, default=None)
    parser.add_argument("--use_global_learning_rate", action="store_true", help="Use global learning rate for opt")
    parser.add_argument("--seed", type=int, default=444, help="The seed for the experiment")    
    parser.add_argument(
        "--max_micro_batch_size",
        type=int,
        default=None,
        help="The max micro batch size for the base scale"
    )    
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="The micro batch size for the base scale"
    )    
    parser.add_argument(
        "--lr_schedule_path",
        type=str,
        default=None,
        help="The path to the LR schedule yaml file"
    )
    parser.add_argument(
        "--slurm_partition",
        type=str,
        default="rtx-2080",
        choices=["rtx-2080", "l40"],
        help="The SLURM partition to use for the experiment"
    )

    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Activate DDP explicitly."
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Devices to pick up for training."
    )    
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Devices to pick up for training."
    )
    parser.add_argument(
        "--ddp_strategy",
        type=str,
        default="ddp",
        choices=["ddp", "ddp2", "ddp_spawn", "deepspeed"],
        help="The DDP strategy to use. Requires `--ddp` to be set."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # if hasattr(args, "base_lr"):
    #     assert hasattr(args, "mup_base"), "MuP base file path is required for when using base LR."
    if hasattr(args, "mup_base"):
        assert hasattr(args, "base_lr"), "Base LR is required for when using MuP base file."

    # Setting experiment canvas for path management
    canvas = ExpCanvas(CANVAS_BASE_PATH, args.canvas_access)

    # Loading
    yaml.SafeLoader.add_constructor('!path', path_constructor)
    with (canvas.train_template if args.train_template is None else Path(args.train_template)).open(
            encoding="utf-8") as yaml_file:
        train_config = yaml.safe_load(yaml_file)
    if args.lr_schedule_path:
        with Path(args.lr_schedule_path).open(encoding="utf-8") as yaml_file:
            lr_schedule = yaml.safe_load(yaml_file)
    
    if args.micro_batch_size is not None:   
        train_config["micro_batch_size"] = args.micro_batch_size
    if args.max_micro_batch_size is not None:
        train_config["max_micro_batch_size"] = args.max_micro_batch_size
    if args.target_scale is not None:
        with (Path(args.target_scale)).open(encoding="utf-8") as yaml_file:
            model_config = yaml.safe_load(yaml_file)
        if "max_micro_batch_size" in model_config:
            _max_micro_batch_size = model_config.pop("max_micro_batch_size")
            if args.max_micro_batch_size is not None:
                train_config["max_micro_batch_size"] = args.max_micro_batch_size
                print("WARNING: `max_micro_batch_size` provided; overriding model `max_micro_batch_size`")
            else:
                if _max_micro_batch_size is not None:
                    if isinstance(_max_micro_batch_size, dict):
                        train_config["max_micro_batch_size"] = _max_micro_batch_size[args.slurm_partition]
                    elif isinstance(_max_micro_batch_size, int):
                        train_config["max_micro_batch_size"] = _max_micro_batch_size
                    else:
                        raise ValueError("Invalid max_micro_batch_size value")
        
        train_config["model_config"] = model_config
        train_config["block_size"] = model_config["block_size"]

    # adjusting for muP
    if args.base_lr is not None:
        train_config["max_lr"] = args.base_lr  # crucial for muP to work properly
    if args.mup_base is not None:
        train_config["mup_base_shape_path"] = Path(args.mup_base)

    # adjusting for global learning rate
    if args.use_global_learning_rate:
        train_config["use_global_learning_rate"] = True

    # adjusting for warmstarting
    train_config = warmstart_parser(args, train_config)

    # adjusting LR schedule, if provided
    if args.lr_schedule_path:
        for key, value in lr_schedule.items():
            train_config[key] = value

    train_config["seed"] = args.seed
    
    _strategy = args.ddp_strategy if args.ddp else "auto"
    fabric = L.Fabric(
        accelerator="auto",
        devices="auto" if args.devices is None else int(args.devices),
        # num_nodes=args.num_nodes,
        strategy=_strategy
    )
    train_config.update({"devices": fabric.world_size})

    print("=" * 50)
    print(vars(fabric))
    print("=" * 50)

    train_config = TrainConfig(**train_config)
    data_config = prepare_data_handler_from_file(
        data_config_path=canvas.data_handler_root / DATASET_MAP(args.dataset),
        train_config=train_config,
        root_data_path=canvas.data_root
    )

    # Running
    result_dict = main(
        fabric=fabric,
        data=data_config,
        train_args=train_config,
        out_dir=canvas.results_root / args.output_tree / f"seed={args.seed}"  # uses the canvas info as parent directory
    )
# end of file
