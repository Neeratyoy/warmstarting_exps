from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from litgpt.utils import num_parameters
from litgpt.config import Config
from saws.model import GPT_Scales

SCALE_CONFIG_FILE = lambda block, depth, scale: (
    f"width-only_block={block}_depth={depth}_scale{scale}.yaml"
)


def convert_color_tuple_to_hex(color_tuple: tuple[int | float, int | float, int | float]) -> str:
    hex_color = "#"
    for color in color_tuple:
        # check if color is a float
        if isinstance(color, float):
            # convert float to int
            color = int(color * 255)
        hex_color += f"{color:02x}"
    return hex_color

def get_number_of_model_parameters(model_root: Path, block: int, depth: int, scale: int) -> int:
    model_config_file = model_root / SCALE_CONFIG_FILE(block, depth, scale)
    with model_config_file.open(encoding="utf-8") as yaml_file:
        model_config = yaml.safe_load(yaml_file)
        model_config.pop("max_micro_batch_size")
    model_config = Config(**model_config)
    return num_parameters(GPT_Scales(model_config), requires_grad=True)

def calculate_token_per_param(tokens_per_param_target_model: int | float,
                              block: int,
                              depth: int,
                              scales: list[int],
                              model_root: Path,
                              include_lowest_scale: bool) -> float:
    """
    We use the same number of tokens per parameter for every model and match the total number of flops
    of the entire warmstarting chain with the total number of flops of the target model.


    We only consider the compute of the lowest scale model if include_lowest_scale is true since the
    computation of this is part of the initial grid search.
    """
    parameters_base_models = []
    lowest_scale = 0 if include_lowest_scale else 1
    for scale in scales[lowest_scale:]:
        parameters_base_models.append(get_number_of_model_parameters(model_root, block, depth, scale) ** 2)

    parameters_target_model = parameters_base_models[-1]
    parameters_base_models_sum = sum(parameters_base_models)

    tokens_per_param = parameters_target_model / parameters_base_models_sum * tokens_per_param_target_model
    return tokens_per_param


if __name__ == "__main__":
    colors = sns.color_palette(plt.cm.magma(np.linspace(0.25, 0.75, 10)))
    colors = [convert_color_tuple_to_hex(color) for color in colors]
    print(colors)
