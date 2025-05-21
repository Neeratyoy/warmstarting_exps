from __future__ import annotations

"""Plot *test‑loss vs. checkpoint step* curves for multiple warm‑starting runs.

This helper is designed to slot straight into the existing *plot_grid* pipeline –
its call signature mirrors `plot_results`, but it consumes the consolidated
`result_test.csv` artefacts instead of TensorBoard logs.

Example
-------
```python
plot_test(
    ax=ax,
    results_dirs=[
        "/…/results/neurips/n_heads/mup/s3-s4",
        "/…/results/neurips/n_heads/warm_auto_snp/s3-s4",
        …
    ],
    run_names=["µP", "Auto‑SNP", …],
    styles=[{"linewidth": 2}, …],
    seeds=[666],              # list[int] – will look in seed=<N>/result_test.csv
    decimals=2,               # round loss for legend hover‑tooltips
)
```
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _read_test_csv(csv_path: Path, metric: str) -> pd.DataFrame:
    """Return a DataFrame with columns *step* and *metric* parsed from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    try:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["step", "filename", "metric_name", metric, "std", "v2", "std2"],
            usecols=[0, 3],
            dtype={0: int, 3: float},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not parse {csv_path}: {exc}") from exc

    df.rename(columns={metric: metric}, inplace=True)
    return df


def _collect_test_data(
    results_dir: Path,
    seeds: List[int],
    metric: str = "loss",
) -> pd.DataFrame:
    """Aggregate test‑loss curves from multiple *seed=<N>* sub‑dirs."""
    parts: list[pd.DataFrame] = []
    for seed in seeds:
        csv_path = results_dir / f"seed={seed}" / "result_test.csv"
        if not csv_path.exists():
            print(f"[warn] {csv_path} not found – skipping seed {seed}")
            continue
        df = _read_test_csv(csv_path, metric)
        df["seed"] = seed
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No result_test.csv files found under {results_dir} for seeds {seeds}.")

    return pd.concat(parts, ignore_index=True)

# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

def plot_test(
    ax: plt.Axes,
    results_dirs: List[str | Path],
    run_names: List[str],
    styles: List[dict],
    seeds: List[int] | None = None,
    x_axis: str = "flops",
    metric: str = "Test Loss",
    smoothing: bool = False,
    smoothing_window: int = 3,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    remove_x_labels: bool = False,
    remove_y_labels: bool = False,
    remove_legend: bool = False,
    x_scale: str = "linear",
    y_scale: str = "linear",
    subtitle: str | None = None,
    **kwargs: dict,
) -> None:
    """Plot *metric* vs *step* curves – one curve per *results_dir*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The host axis to draw on.
    results_dirs, run_names, styles : list‑likes of equal length.
    seeds : list[int]
        Look inside `<results_dir>/seed=<N>/result_test.csv` for each *N*.
    smoothing : bool
        When *True*, apply a centred rolling mean with *smoothing_window* to each
        individual seed curve *before* aggregating (std‑error bands are drawn by
        *seaborn*).
    decimals : int
        Number of decimal places for loss values (displayed in hover‑tooltips).
    """

    if seeds is None:
        seeds = [666]

    if not (len(results_dirs) == len(run_names) == len(styles)):
        raise ValueError("results_dirs, run_names, styles must have matching length")

    results_dirs = [Path(p) for p in results_dirs]

    for run_name, results_dir, style in zip(run_names, results_dirs, styles):
        # ---------------------------------------------------------------------
        # Collect + optional smoothing
        # ---------------------------------------------------------------------
        try:
            df = _collect_test_data(results_dir, seeds, metric)
            if smoothing and smoothing_window > 1:
                df = (
                    df.sort_values("step")
                    .groupby("seed")
                    .apply(lambda g: g.rolling(smoothing_window, on="step", min_periods=1, center=True).mean())
                    .reset_index(drop=True)
                )
        except RuntimeError as exc:
            print(f"[warn] {exc} – skipping {results_dir}")
            continue
        except FileNotFoundError as exc:
            print(f"[warn] {exc} – skipping {results_dir}")
            continue
        except Exception as exc:
            print(f"[warn] {exc} – skipping {results_dir}")
            continue

        # ---------------------------------------------------------------------
        # Style handling – allow caller to override colours / markers, but keep
        # consistency with global palette if desired.
        # ---------------------------------------------------------------------
        if "marker" in style:
            n_points = len(df)
            style.setdefault("markevery", np.linspace(0, n_points - 1, num=12, dtype=int))

        sns.lineplot(
            data=df,
            x="step",
            y=metric,
            hue=None,          # aggregate seeds via seaborn's estimator (mean)
            errorbar="se",     # standard error bands
            label=run_name,
            ax=ax,
            **style,
        )

        ax.scatter(
            data=df,
            x="step",
            y=metric,
            marker="o",
            s=100,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.5,
            color=style.get("color", "black"),
            zorder=10,
        )

    # -------------------------------------------------------------------------
    # Axis / legend housekeeping
    # -------------------------------------------------------------------------
    if remove_x_labels:
        ax.set_xlabel("")
    else:
        ax.set_xlabel(x_axis if x_axis != "flops" else "FLOPs")

    if remove_y_labels:
        ax.set_ylabel("")
    else:
        ax.set_ylabel(metric, fontsize=21)

    if not remove_legend:
        ax.legend(fontsize=20)
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    if subtitle is not None:
        ax.set_title(subtitle, fontsize=20)

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)