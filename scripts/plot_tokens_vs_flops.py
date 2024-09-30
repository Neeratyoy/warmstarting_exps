import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from saws.plot_results import NAMES_DICT, COLOR_DICT
from scripts.utils import calculate_token_per_param, get_number_of_model_parameters
import seaborn as sns
import numpy as np
BASE_MODEL_TOKENS_PER_PARAM = 20


def plot_tokens_vs_flops(
        ax: plt.Axes,
        run_names: list[str],
        scales: list[str],
        warmstart_schedules: list[str],
        remove_x_labels: bool = False,
        remove_y_labels: bool = False,
        remove_legend: bool = False,
        tokens_per_param_target_model: float = 20,
        block: int = 1024,
        depth: int = 6,
):
    assert len(set([scale.split("-")[0] for scale in scales])) == 1, "The first scale of all runs must be the same."
    assert len(set([scale.split("-")[-1] for scale in scales])) == 1, "The last scale of all runs must be the same."
    model_root = Path(__file__).parent.parent / 'configs' / 'width_only' / 'dev'
    base_tokens = BASE_MODEL_TOKENS_PER_PARAM * get_number_of_model_parameters(model_root, block, depth,
                                                                               int(scales[0].split("-")[0]))
    # vertical lines
    vlines_x = []
    colors = sns.color_palette("deep", n_colors=len(run_names))
    # ax.grid(axis='x', linestyle='')
    for idx, (run_name, scale, warmstart_schedule) in enumerate(zip(run_names, scales, warmstart_schedules)):
        scale = [int(s) for s in scale.split("-")]
        if warmstart_schedule == "same_tokens":
            tokens_per_param = calculate_token_per_param(
                tokens_per_param_target_model=tokens_per_param_target_model,
                block=block,
                depth=depth,
                scales=scale,
                model_root=model_root,
                include_lowest_scale=False,
            )
            tokens = [base_tokens]
            compute = [0]
            for s in scale[1:]:
                parameters_current_scale = get_number_of_model_parameters(model_root, block, depth, s)
                tokens_current_scale = tokens_per_param * parameters_current_scale
                compute_current_scale = 6 * tokens_current_scale * parameters_current_scale  # 6N*D for flops
                tokens.append(tokens[-1] + tokens_current_scale)
                compute.append(compute[-1] + compute_current_scale)
        elif warmstart_schedule == "mup":
            tokens = [0]
            compute = [0]
            parameters_current_scale = get_number_of_model_parameters(model_root, block, depth, scale[-1])
            tokens_current_scale = tokens_per_param_target_model * parameters_current_scale
            compute_current_scale = 6 * tokens_current_scale * parameters_current_scale
            tokens.append(tokens[-1] + tokens_current_scale)
            compute.append(compute[-1] + compute_current_scale)
        else:
            raise ValueError(f"Unknown warmstart schedule: {warmstart_schedule}")

        if run_name in COLOR_DICT:
            colors[idx] = COLOR_DICT[run_name]
        if run_name in NAMES_DICT:
            run_name = NAMES_DICT[run_name]

        sns.lineplot(
            x=tokens,
            y=compute,
            label=run_name,
            ax=ax,
            marker='o',
            markevery=1,
            zorder=2,
            color=colors[idx],
        )

        vlines_x.append(tokens[-1])

    if not remove_x_labels:
        ax.set_xlabel("tokens")
    if not remove_y_labels:
        ax.set_ylabel("FLOPs")
    if not remove_legend:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)
    else:
        ax.get_legend().remove()

    sns.despine(ax=ax, top=True, right=True)

    ax.axhline(y=compute[-1], color='black', linestyle='-', linewidth=2, zorder=1)
    ax.text(x=ax.get_xlim()[1], y=compute[-1]*1.025, s='max FLOPs', color='black', ha='right', va='bottom')

    y_lim = ax.get_ylim()
    ax.vlines(x=vlines_x, ymin=ax.get_ylim()[0], ymax=compute[-1], color=colors, linestyle=':', linewidth=2, zorder=0)
    ax.set_ylim(y_lim)

def get_args():
    parser = argparse.ArgumentParser(
        description="Plot the number of tokens per parameter vs the number of FLOPs for different warmstarting methods."
    )
    parser.add_argument(
        "--run_names",
        type=str,
        nargs="+",
        required=True,
        help="The names of the runs corresponding to the different warmstarting scales and schedules.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        required=True,
        help="The scales of the different warmstarting runs. Each scale must be seperated by a '-'. For example: '0-2-4'.",
    )
    parser.add_argument(
        "--warmstart_schedules",
        type=str,
        nargs="+",
        required=True,
        choices=["same_tokens", "mup"],
        help=("The algorithm used to determine the compute spend for each scale. "
              "The number of scales must match the number of warmstart schedules."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the figueres are saved.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="The filename of the saved figure.",
    )
    parser.add_argument(
        "--tokens_per_param_target_model",
        type=float,
        default=20,
        help="The number of tokens per parameter for the target model.",
    )
    parser.add_argument(
        "--block",
        default=1024,
        help="The block size of the model.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="The depth of the model.",
    )
    args = parser.parse_args()
    assert len(args.run_names) == len(args.scales) == len(args.warmstart_schedules), (
        "The number of run names, scales and warmstart schedules must match."
    )
    return args


if __name__ == "__main__":
    args = get_args()

    sns.set_style("white")
    sns.set_context("talk")
    fig, ax = plt.subplots()

    plot_tokens_vs_flops(
        ax=ax,
        run_names=args.run_names,
        scales=args.scales,
        warmstart_schedules=args.warmstart_schedules,
        tokens_per_param_target_model=args.tokens_per_param_target_model,
        block=args.block,
        depth=args.depth,
    )
    output_dir = Path(args.output_dir)
    plt.tight_layout()
    plt.show()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{args.file_name}.png")
    fig.savefig(output_dir / f"{args.file_name}.pdf")
    plt.close(fig)