import argparse
import functools
import copy
from pathlib import Path
from saws.plot_results import plot_results
import yaml
from matplotlib import pyplot as plt
import seaborn as sns

def plot_grid(config_file: str | Path):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    default_axes_args = config["default_axes_args"]
    global_legend = 'global_legend' in config and config['global_legend']
    global_xlabel = config['global_xlabel']
    global_ylabel = config['global_ylabel']

    figsize = (config['ncols'] * 4, config['nrows'] * 3)
    fig = plt.figure(layout='constrained', figsize=figsize)
    gs = fig.add_gridspec(config['nrows'], config['ncols'], wspace=0.05, hspace=0.05)
    sns.set_context(config['sns_context'])

    axes = {}
    for ax_key, ax_config in config['axes'].items():
        row, col = ax_key.split(",")
        if ":" in row:
            row = [int(r) for r in row.split(":")]
        else:
            row = int(row), int(row) + 1
        if ":" in col:
            col = [int(c) for c in col.split(":")]
        else:
            col = int(col), int(col) + 1

        sharex = None
        sharey = None
        if 'sharex' in ax_config and ax_config['sharex'] is not None:
            sharex = axes[ax_config['sharex']]
        if 'sharey' in ax_config and ax_config['sharey'] is not None:
            sharey = axes[ax_config['sharey']]

        ax = fig.add_subplot(gs[row[0]:row[1], col[0]:col[1]], sharex=sharex, sharey=sharey)
        axes[ax_key] = ax

        if 'styles' not in ax_config["plotting_function_args"]:
            ax_config["plotting_function_args"]["styles"] = [{} for _ in range(
                len(ax_config["plotting_function_args"]["results_dirs"]))]

        plotting_function_args = copy.deepcopy(default_axes_args)
        plotting_function_args.update(ax_config["plotting_function_args"])

        if global_legend:
            plotting_function_args['remove_legend'] = True
        if global_xlabel is not None:
            plotting_function_args['remove_x_labels'] = True
        if global_ylabel is not None:
            plotting_function_args['remove_y_labels'] = True

        plot_results(ax=ax, **plotting_function_args)

    if global_xlabel is not None:
        fig.supxlabel(global_xlabel)
    if global_ylabel is not None:
        fig.supylabel(global_ylabel)
    if global_legend:
        legend_handels_labels = [list(zip(*ax.get_legend_handles_labels())) for ax in axes.values()]
        legend_handels_labels = functools.reduce(lambda a, b: a+b, legend_handels_labels)
        unique = dict([(label, handle) for (handle, label) in legend_handels_labels])
        fig.legend(unique.values(), unique.keys(), loc='outside upper center', ncol=8, borderaxespad=0.1) 

    plt.show()
    plt.close(fig)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_dir / f"{config['file_name']}.png")
    fig.savefig(output_dir / f"{config['file_name']}.pdf")
    # determine the grid size


def get_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The yaml file definign the plotting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    plot_grid(
        config_file=args.config_file,
    )
