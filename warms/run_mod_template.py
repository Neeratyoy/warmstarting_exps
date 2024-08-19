import argparse
from copy import deepcopy
import lightning as L
from litgpt.config import Config
from pathlib import Path

from saws import DataHandler, TrainConfig, main
from saws.config.utils import preprocess_wikitext

from warms.utils.support import prepare_data_handler_from_file


BASE_PATH = Path("/work/dlclarge1/mallik-warmstarting")
DATA_BASE_PATH = BASE_PATH / "warmstarting_exps" / "configs" / "data_handlers"
TEMPLATE_PATH = BASE_PATH / "warmstarting_exps" / "configs" / "train_template.yaml"


def get_args():
    parser = argparse.ArgumentParser(description="Parser for generating MuP base files")

    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset choice",
        choices=["wikitext", "slimpajama"]
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/work/dlclarge1/mallik-warmstarting/scale_and_warmstart/data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=BASE_PATH / "warmstarting_exps" / "results" / "temp",
    )
    parser.add_argument(
        "--mup_base",
        type=str,
        required=True,
        help="The path to the .bsh file for base muP scale"
    )   
    parser.add_argument(
        "--mup_target",
        type=str,
        required=True,
        help="The path to target scale model config"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Loading
    train_config = TrainConfig.from_path(TEMPLATE_PATH)
    _train_config = deepcopy(train_config)
    data_config = prepare_data_handler_from_file(
        DATA_BASE_PATH / f"{args.dataset}_data_handler.yaml",
        train_config,
        Path(args.data_root_path),
    )
    model_config = Config.from_file(Path(args.mup_target))

    # Updating
    for k, v in train_config.model_config.to_dict().items():
        if k in model_config.__dict__:
            setattr(train_config.model_config, k, getattr(model_config, k, v))
    train_config.model_config.d_model = model_config.n_embd
    train_config.block_size = model_config.block_size
    train_config.mup_base_shape_path = Path(args.mup_base)
    
    # Running
    fabric = L.Fabric(devices="auto", strategy="auto")
    main(
        fabric,
        data_config,
        train_config,
        Path(args.output_dir),
    )
