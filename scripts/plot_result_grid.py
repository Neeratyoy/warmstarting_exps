import argparse
import functools
import copy
from pathlib import Path
from saws.plot_results import plot_results
import yaml
from matplotlib import pyplot as plt
import seaborn as sns

from scripts.plot_successive_schedule import plot_warmstarting_schedules
from scripts.plot_tokens_vs_flops import plot_tokens_vs_flops


def plot_grid(config_file: str | Path):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    default_axes_args = config["default_axes_args"]
    global_legend = 'global_legend' in config and config['global_legend']
    global_xlabel = config['global_xlabel']
    global_ylabel = config['global_ylabel']

    figsize = (config['ncols'] * 4, config['nrows'] * 3)
    if 'ncols_plots' in config and config['ncols_plots'] is not None:
        figsize = config['ncols_plots'] * 4, figsize[1]
    if 'nrows_plots' in config and config['nrows_plots'] is not None:
        figsize = figsize[0], config['nrows_plots'] * 3
    
    if global_legend:
        figsize = figsize[0], figsize[1] + 0.5
    
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

        plotting_function_args = {}
        if 'function' not in ax_config or ax_config['function'] == 'plot_results':
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

        print(f"Plotting {ax_key}")

        if 'function' not in ax_config or ax_config['function'] == 'plot_results':
            plot_results(ax=ax, **plotting_function_args)
        elif ax_config['function'] == 'plot_tokens_vs_flops':
            plot_tokens_vs_flops(ax=ax, **plotting_function_args)
        elif ax_config['function'] == 'plot_warmstarting_schedules':
            plot_warmstarting_schedules(ax=ax, **plotting_function_args)

    if global_xlabel is not None:
        # fig.supxlabel(global_xlabel)
        _label = fig.supxlabel(global_xlabel)
        _label.set_fontsize(28)
    if global_ylabel is not None:
        fig.supylabel(global_ylabel)
    if global_legend:
        legend_handels_labels = [list(zip(*ax.get_legend_handles_labels())) for ax in axes.values()]
        legend_handels_labels = functools.reduce(lambda a, b: a + b, legend_handels_labels)
        unique = dict([(label, handle) for (handle, label) in legend_handels_labels])
        
        if 'global_xlabel' not in config or config['global_xlabel'] is None:
            fig.legend(unique.values(), unique.keys(), loc='outside lower center', ncol=config['global_legend_ncols'], 
                    borderaxespad=0.1)
        else:
            fig.legend(unique.values(), unique.keys(), loc='outside upper center', ncol=config['global_legend_ncols'],
                    borderaxespad=0.1)
    ##########
    # WARNING:
    # custom temp
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(-1.5, -1.5), ncol=5, fontsize=25)

    plt.show()
    plt.close(fig)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(output_dir / f"{config['file_name']}.png")
    fig.savefig(output_dir / f"{config['file_name']}.pdf")

    print(f"Saved figure to {output_dir / config['file_name']}.png")
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
