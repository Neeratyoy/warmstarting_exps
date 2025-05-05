import argparse
from pathlib import Path

from saws.config.data_config import DataHandler, preprocess_wikitext
from saws.config.train_config import TrainConfig


def prepare_data_handler_from_file(
    data_config_path: Path,
    train_config: TrainConfig,
    root_data_path: Path | None = None,
) -> DataHandler:
    """Load a DataHandler object from a YAML file.

    Args:
    data_config_path (Path): Path to the YAML file containing the DataHandler configuration
    train_config: TrainConfig object

    Returns:
    DataHandler: DataHandler object
    """
    def _postprocess_data_handler(
        data_config: DataHandler,
        root_data_path: str | Path | None = None,
        seed: int | None = None,
        block_size: int = 1024,
    ) -> DataHandler:
        data_config.preprocess_fn = preprocess_wikitext
        data_config.root_data_path = (
            Path(root_data_path) 
            if root_data_path is not None 
            else Path("./").absolute() / "data"
        )
        data_config.seed = seed if seed is not None else data_config.seed
        data_config.block_size = block_size
        return data_config
   
    data_config = DataHandler.from_path(data_config_path)
    if data_config.hf_dataset_id is None:
        data_config.seed = train_config.seed
        data_config.root_data_path = root_data_path / data_config.root_data_path
        return data_config

    data_config = _postprocess_data_handler(
        data_config,
        root_data_path,
        train_config.seed,
        train_config.block_size
    )

    return data_config


def warmstart_parser(args: argparse.Namespace, train_config: dict) -> dict:
    if args.warmstart:
        train_config["warmstart_config"]["activate"] = True
        if args.warmstart_type is not None:
            train_config["warmstart_config"]["warmstart_type"] = args.warmstart_type

        if args.base_model_step is not None:
            if "warmstarting_args" not in train_config["warmstart_config"]:
                train_config["warmstart_config"]["warmstarting_args"] = {}
            train_config["warmstart_config"]["warmstarting_args"]["base_model_step"] = args.base_model_step

        if args.warmstart_base_path is not None:
            train_config["warmstart_config"]["base_model_path"] = args.warmstart_base_path
        else:
            assert train_config["warmstart_config"][
                       "base_model_path"] is not None, "Base model path is required for warmstarting."

        # Hyperparameters for warmstarting methods
        if args.shrinking_factor is not None:
            if "warmstarting_args" not in train_config["warmstart_config"]:
                train_config["warmstart_config"]["warmstarting_args"] = {}
            train_config["warmstart_config"]["warmstarting_args"]["shrinking_factor"] = args.shrinking_factor
        if args.perturbation_sigma is not None:
            if "warmstarting_args" not in train_config["warmstart_config"]:
                train_config["warmstart_config"]["warmstarting_args"] = {}
            train_config["warmstart_config"]["warmstarting_args"]["perturbation_sigma"] = args.perturbation_sigma

    return train_config