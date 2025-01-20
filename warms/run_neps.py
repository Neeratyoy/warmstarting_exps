"""
This script is used to run NePS based on the given train template, dataset and NePS yaml configuration.

Example usage for running a grid search on the learning rate for the slimpajama dataset:

python warms/run_neps.py \
    --neps_config_path <root_path>/configs/neps/lr_grid_search.yaml \
    --dataset slimpajama \
    --output_tree neps/quick_debug \
    --neps_seed 123 \
    --grid_search \ # Only set this if you are using grid search!
    --target_scale <path_to_target_scale_config>

```lr_grid_search.yaml
pipeline_space:
  max_lr:
    choices: [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
max_evaluations_total: 6
```

If multi-fidelity is being used, the runtime parameter used as fidelity needs to provided as a constant
to indicate the total runtime. Additionally, it needs to be provided as a hyperparameter while using the prefix
`early_stopping_` to control the fidelity brackets. This is needed for the lr scheduling to work correctly.
Note that you can provide other attributes as constants in the yaml file as well.

See the example below for a multi-fidelity hyperband search:
```
pipeline_space:
  max_lr:
    lower: 0.0001
    upper: 0.03
    log: True
  tokens_per_param: 20
  early_stopping_tokens_per_param:
    lower: 2.5
    upper: 20.0
    is_fidelity: true
max_evaluations_total: 100
searcher:
  strategy: "hyperband"
  eta: 2
```

Here, we are able to specify the search algorithm in the yaml file while we had pass it as an argument when using
grid search. This is because grid search is not natively supported in NePS.
"""

import argparse
from functools import reduce
from pathlib import Path
from typing import Callable

import neps
import numpy as np
import pandas as pd
import torch
import yaml
import random

from warms import (
    CANVAS_BASE_PATH,
    DATASET_MAP,
    ExpCanvas,
    prepare_data_handler_from_file
)
import lightning as L
from saws.config.yaml_utils import path_constructor
from saws import TrainConfig, main
from neps.optimizers.grid_search.optimizer import GridSearch

from warms.utils.support import warmstart_parser

from scale_and_warmstart.saws.pretrain import train


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="Parser for ")

    parser.add_argument(
        "--neps_config_path",
        type=str,
        required=True,
        help="The path to config yaml file that defines the search space, searcher, and other neps arguments",
    )
    parser.add_argument(
        "--train_template",
        type=str,
        help="Path to the train config template file. "
             "If not provided, the default template in the selected canvas will be used.",
    )
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
        choices=["wikitext", "slimpajama"]
    )
    parser.add_argument(
        "--output_tree",
        type=str,
        help="Creates a subdirectory tree starting from `results_root` in canvas configuration and uses it as the neps output directory. "
             "If not provided the `root_directory` in the config .yaml will be used. If both are given, an error will be raised.",
    )
    parser.add_argument(
        "--target_scale",
        type=str,
        help="Path to the config file for the model at the target scale. "
             "If not provided, the model config from the train template will be used.",
    )

    # Warmstarting arguments
    parser.add_argument(
        "--warmstart",
        action="store_true",
        help="Set to true to overwrite the warmstarting values provided by train_template with argparse arguments.")
    parser.add_argument("--warmstart_type", type=str)
    parser.add_argument("--warmstart_base_path", type=str)
    parser.add_argument("--base_model_step", type=int)
    parser.add_argument("--shrinking_factor", type=float)
    parser.add_argument("--perturbation_sigma", type=float)

    parser.add_argument(
        "--warmstart_neps_root_path",
        type=str,
        help="Specific to one kind of experiment: If you do a grid search and want to warmstart the model with a model "
             "from a different grid search that has the same hyperparameters."
             "Provide the root path of the neps output directory of the gridsearch you want to warmstart from.")

    parser.add_argument(
        "--mup_base_path",
        type=str,
        help="The path to the .bsh file for base muP scale. If not provided, sp will be used.",
    )
    parser.add_argument(
        "--neps_seed",
        type=int,
        default=123,
        help="The seed used by the neps optimizer.",
    )
    # Grid search is not a valid literal as a searcher for neps 12.2 and can't be specified in the yaml so it has to be passed as an argument.
    parser.add_argument('--grid_search',
                        action='store_true',
                        help='Does grid search instead of the default search.')

    parser.add_argument(
        "--slurm_partition",
        type=str,
        default="rtx-2080",
        help="The SLURM partition to use for the experiment"
    )
    args = parser.parse_args()

    if (args.warmstart_type is not None
            or args.warmstart_base_path is not None
            or args.base_model_step is not None
            or args.shrinking_factor is not None
            or args.perturbation_sigma is not None
            or args.warmstart_neps_root_path is not None):
        assert args.warmstart, "Warmstart arguments are provided but warmstart is not set to True."

    return args


def warmstart_from_neps(train_config: dict, warmstart_neps_root_path: str):
    """
    Modifies the train_config in such a way that. If the search spaces do not align there is a lot that can go wrong.
    :param train_config:
    :param config:
    :param warmstart_neps_root_path:
    """
    if warmstart_neps_root_path is not None:
        summary_path = Path(warmstart_neps_root_path) / "summary_csv" / "config_data.csv"
        if not summary_path.exists():
            raise ValueError(
                "The summary .csv file does not exist. This might be because the NePS run did not finish successfully "
                "for all configurations.")

        summary = pd.read_csv(summary_path)

        # Check if all the runs finished successfully, if not raise an error
        if not all(summary["status"] == "complete"):
            raise ValueError("Not all runs in the neps output directory finished successfully. "
                             "Please make sure all runs are completed before using warmstart.")

        # select all the columns that are hyperparameters (more than 1 unique value and start with "config.")
        hp_columns = [col.removeprefix("config.") for col in summary.columns if
                      col.startswith("config.") and summary[col].nunique() > 1]

        # select the row that matches the train_config hp values
        row_selection = [summary["config." + col] == train_config[col] for col in hp_columns]
        row_selection = reduce(lambda x, y: x & y, row_selection)
        config = summary["config_id"][row_selection]

        assert len(config) == 1, "The warmstart config could not be found or might not be unique."
        config = config.iloc[0]

        train_config["warmstart_config"]["base_model_path"] = Path(
            warmstart_neps_root_path) / "results" / f"config_{config}" / "output"


def neps_training_wrapper(args: argparse.Namespace) -> Callable:
    """
    Wrapper function to create a pipeline for neps training
    :param args: argparse arguments
    :return: function that can be passed to neps.run for the run_pipeline argument
    """

    def run_pipeline(pipeline_directory: Path, previous_pipeline_directory: Path, **config) -> dict:
        """
        Runs the pipeline with the given neps configuration and the arguments from the wrapper function.
        Supports resuming the pipeline from a previous pipeline directory to speed up multi-fidelity HPO.

        :param pipeline_directory: The directory where the information of the current pipeline is saved.
        :param previous_pipeline_directory: The directory from which the pipeline will be resumed if multifidelity is used.
        :param config: The hyperparameters and constants to be used in the pipeline.
        :return: loss at the end of training
        """
        canvas = ExpCanvas(CANVAS_BASE_PATH, args.canvas_access)

        # Load the config from the train_template as base
        yaml.SafeLoader.add_constructor('!path', path_constructor)
        with (canvas.train_template if args.train_template is None else Path(args.train_template)).open(
                encoding="utf-8") as yaml_file:
            train_config = yaml.safe_load(yaml_file)

        # Use model config from target_scale path if given
        if args.target_scale is not None:
            with (Path(args.target_scale)).open(encoding="utf-8") as yaml_file:
                model_config = yaml.safe_load(yaml_file)
            if "max_micro_batch_size" in model_config:
                _max_micro_batch_size = model_config.pop("max_micro_batch_size")
                if isinstance(_max_micro_batch_size, dict):
                    train_config["max_micro_batch_size"] = _max_micro_batch_size[args.slurm_partition]
                elif isinstance(_max_micro_batch_size, int):
                    train_config["max_micro_batch_size"] = _max_micro_batch_size
                else:
                    raise ValueError("Invalid max_micro_batch_size value")

            train_config["model_config"] = model_config
            train_config["block_size"] = model_config["block_size"]

        train_config["mup_base_shape_path"] = Path(args.mup_base_path) if args.mup_base_path is not None else None

        train_config = warmstart_parser(args, train_config)

        # Apply all the hyperparameters and constants from the config to the train_config
        for key, value in config.items():
            # Nested dictionaries can be specified by combining the keys of the nested dict with a '.'
            sub_dict = train_config  # this is the innermost dictionary which key and value will be set
            for nested_key in key.split('.')[:-1]:  # we are iterating through the keys of the nested dictionary
                if nested_key not in sub_dict or not isinstance(sub_dict[nested_key], dict):
                    sub_dict[nested_key] = {}
                sub_dict = sub_dict[nested_key]

            # Setting all training time arguments except the given one to None since we can't pass None constants via NePS.
            trainings_time_arguments = ["tokens_per_param", "max_tokens", "max_train_steps"]
            if key in trainings_time_arguments:
                for tt_arg in trainings_time_arguments:
                    sub_dict[tt_arg] = None

            sub_dict[key.split('.')[-1]] = value

        # Resume run from previous fidelity
        if previous_pipeline_directory is not None:
            train_config["load_state_path"] = previous_pipeline_directory / "output"

        warmstart_from_neps(train_config, args.warmstart_neps_root_path)

        train_config = TrainConfig(**train_config)

        data_config = prepare_data_handler_from_file(
            data_config_path=canvas.data_handler_root / DATASET_MAP(args.dataset),
            train_config=train_config,
            root_data_path=canvas.data_root
        )

        fabric = L.Fabric(devices="auto", strategy="auto")
        run_metrics = main(
            fabric=fabric,
            data=data_config,
            train_args=train_config,
            out_dir=Path(pipeline_directory) / "output"
        )
        return {"loss": run_metrics["val_loss"], "info_dict": run_metrics}

    return run_pipeline


if __name__ == "__main__":
    args = get_args()

    set_seed(args.neps_seed)

    optional_args = {}
    if args.output_tree is not None:
        canvas = ExpCanvas(CANVAS_BASE_PATH, args.canvas_access)
        optional_args["root_directory"] = canvas.results_root / args.output_tree
    if args.grid_search:
        optional_args["searcher"] = GridSearch

    neps.run(
        run_pipeline=neps_training_wrapper(args),
        run_args=args.neps_config_path,
        **optional_args
    )
