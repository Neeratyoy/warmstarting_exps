from pathlib import Path

from .utils.experiment_canvas import ExpCanvas
from .utils.support import prepare_data_handler_from_file


CANVAS_BASE_PATH = Path(__file__).absolute().parent / ".." / "configs" / "meta_exp_canvas.toml"
DATASET_MAP = lambda x: f"{x}_data_handler.yaml"


__all__ = [
    "ExpCanvas",
    "prepare_data_handler_from_file",
    CANVAS_BASE_PATH,
    DATASET_MAP,

]