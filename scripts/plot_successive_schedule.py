import argparse
from pathlib import Path
import seaborn as sns
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from saws.config.yaml_utils import path_constructor
from saws.plot_results import COLOR_DICT, NAMES_DICT


def load_schedule(result_dir: Path | str, include_smallest_scale: bool = False) -> pd.DataFrame:
    result_dir = Path(result_dir)

    yaml.SafeLoader.add_constructor('!path', path_constructor)

    with (result_dir / "result.yaml").open(encoding="utf-8") as yaml_file:
        result = yaml.safe_load(yaml_file)
    train_steps = result["train_steps"]

    with (result_dir / "info.yaml").open(encoding="utf-8") as yaml_file:
        info = yaml.safe_load(yaml_file)
    parameters = info["parameters"]

    with (result_dir / "train_config_post_init.yaml").open(encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
    tokens_per_param = config["tokens_per_param"]

    schedule = pd.DataFrame(
        {"train_steps": [train_steps], "parameters": [parameters], "tokens_per_param": [tokens_per_param]})

    if "warmstart_config" in config and config["warmstart_config"]["activate"]:
        warm_schedule = load_schedule(config["warmstart_config"]["base_model_path"], include_smallest_scale)
        schedule = pd.concat([warm_schedule, schedule], ignore_index=True).reset_index(drop=True)
    elif not include_smallest_scale:
        schedule = pd.DataFrame()

    return schedule


def plot_warmstarting_schedules(
        ax: plt.Axes,
        results_dirs: list[Path | str],
        run_names: list[str],
        y_metric: str,
        include_smallest_scale: bool = False,
        remove_x_labels: bool = False,
        remove_y_labels: bool = False,
        remove_legend: bool = False,
        x_limits: tuple[float, float] | None = None,
        y_limits: tuple[float, float] | None = None,
        y_scale: str = "linear",
        x_scale: str = "linear",
        subtitle: str | None = None,
        **kwargs,
):
    colors = sns.color_palette("deep", n_colors=len(run_names))
    for idx, (result_dir, run_name) in enumerate(zip(results_dirs, run_names)):
        if run_name in COLOR_DICT:
            colors[idx] = COLOR_DICT[run_name]
        if run_name in NAMES_DICT:
            run_name = NAMES_DICT[run_name]
        schedule = load_schedule(result_dir, include_smallest_scale)
        sns.lineplot(data=schedule, x="parameters", y=y_metric, ax=ax, label=run_name, color=colors[idx], marker='o')


    if remove_x_labels:
        ax.set_xlabel("")
    if remove_y_labels:
        ax.set_ylabel("")
    if remove_legend:
        ax.get_legend().remove()
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if subtitle is not None:
        ax.set_title(subtitle)
    ax.set_yscale(y_scale)
    ax.set_xscale(x_scale)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dirs",
                        type=str, required=True,
                        nargs='+',
                        help=("The list of directories which will be plotted. Each directory must contain warmstarted "
                              "run directories whose names all end in with the step they were warmstarted from."))
    parser.add_argument("--run_names",
                        type=str, required=True,
                        nargs='+',
                        help="The names of the runs corresponding to the results directories.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output directory where the figures will be saved.")
    parser.add_argument("--y_metric",
                        type=str,
                        choices=["train_steps", "tokens_per_param"],
                        help="The y-axis metric that will be plotted.")
    parser.add_argument("--include_smallest_scale",
                        action="store_true",
                        help="Includes the smallest scale were grid search was done in the plot.")

    args = parser.parse_args()
    assert len(args.run_names) == len(
        args.results_dirs), "The number of run names must match the number of results directories."

    return args


if __name__ == "__main__":
    args = get_args()

    fig, ax = plt.subplots()

    plot_warmstarting_schedules(
        ax,
        args.results_dirs,
        args.run_names,
        args.y_metric,
        args.include_smallest_scale,
    )

    plt.show()
    # plt.savefig(args.output_dir / "warmstarting_schedules.png")