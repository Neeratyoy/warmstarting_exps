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
    data_config = _postprocess_data_handler(
        data_config,
        root_data_path,
        train_config.seed,
        train_config.block_size
    )

    return data_config