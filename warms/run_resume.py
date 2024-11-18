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


def update_config(
    path: Path,
    tokens_per_param: int = None,
    max_tokens: int = None,
    max_train_steps: int = None,
    overwrite: bool = False,
) -> TrainConfig:
    config = TrainConfig.from_path(path / "train_config_post_init.yaml")

    config.save_state_path = deepcopy(path)
    config.load_state_path = deepcopy(path) if not overwrite else None
    if tokens_per_param is not None:
        if tokens_per_param < config.tokens_per_param:
            raise ValueError(f"tokens_per_param must be greater than or equal to {config.tokens_per_param}")
        config.tokens_per_param = tokens_per_param
    if max_tokens is not None:
        if max_tokens < config.max_tokens:
            raise ValueError(f"max_tokens must be greater than or equal to {config.max_tokens}")
        config.max_tokens = max_tokens
    if max_train_steps is not None:
        if max_train_steps < config.max_train_steps:
            raise ValueError(f"max_train_steps must be greater than or equal to {config.max_train_steps}")
        config.max_train_steps = max_train_steps
    if overwrite:
        config.load_state_path = None

    return config


def main(
    path: Path,
    tokens_per_param: int,
    max_tokens: int,
    max_train_steps: int,
    overwrite: bool,
    dataset: str = "slimpajama",
    canvas_access: str = "global",
):
    assert (
        (tokens_per_param or 0) +
        (max_tokens or 0) +
        (max_train_steps or 0)
    ), "One of tokens_per_param, max_tokens, or max_train_steps must be provided."

    assert (
        bool(tokens_per_param or 0) +
        bool(max_tokens or 0) +
        bool(max_train_steps or 0)
    ) == 1, "Only one of tokens_per_param, max_tokens, or max_train_steps must be provided."

    train_config = update_config(
        path=Path(path),
        tokens_per_param=tokens_per_param,
        max_tokens=max_tokens,
        max_train_steps=max_train_steps,
        overwrite=overwrite,
    )

    # Setting experiment canvas for path management
    canvas = ExpCanvas(CANVAS_BASE_PATH, canvas_access)
    data_config = prepare_data_handler_from_file(
        data_config_path=canvas.data_handler_root / DATASET_MAP(dataset),
        train_config=train_config,
        root_data_path=canvas.data_root
    )

    # Running
    fabric = L.Fabric(devices="auto", strategy="auto")
    result_dict = main(
        fabric=fabric,
        data=data_config,
        train_args=train_config,
        out_dir=train_config.save_state_path
    )
    return result_dict


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
        "--path",
        type=str,
        required=True,
        help="Path to the experiment directory to be resumed or retrained."
    )
    parser.add_argument(
        "--tokens_per_param",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the existing run and start training from scratch."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    main(
        path=args.path,
        tokens_per_param=args.tokens_per_param,
        max_tokens=args.max_tokens,
        max_train_steps=args.max_train_steps,
        overwrite=args.overwrite,
        dataset=args.dataset,
        canvas_access=args.canvas_access,
    )
# end of file