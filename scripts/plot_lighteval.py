from __future__ import annotations

"""Plot LightEval downstream‑accuracy curves.

Features
~~~~~~~~
* **aggregate_tasks=True** (default) – average over tasks → one curve per
  *method* (with stderr error bars).
* **aggregate_tasks=False** – one curve per *task* (hue‑coloured) for each
  method.
* **task_filter="<task_name>"** – plot only that task (overrides
  `aggregate_tasks`).
"""

from pathlib import Path
from typing import List, Sequence

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_eval_csv(csv_path: Path) -> pd.DataFrame:
    """Load LightEval CSV, clean up, and return valid rows only."""
    df = pd.read_csv(csv_path, keep_default_na=True)

    # ── Basic cleanup ────────────────────────────────────────────────────
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)

    # Drop spurious rows where the checkpoint column is the placeholder
    # string "lit_model.pth" (no step suffix).
    if "checkpoint" in df.columns:
        df = df[df["checkpoint"] != "lit_model.pth"].copy()

    # If acc_norm is missing but raw acc present, fall back.
    if {"acc", "acc_norm"}.issubset(df.columns):
        missing_norm = df["acc_norm"].isna() & df["acc"].notna()
        df.loc[missing_norm, "acc_norm"] = df.loc[missing_norm, "acc"]
        if "acc_stderr" in df.columns and "acc_norm_stderr" in df.columns:
            df.loc[missing_norm, "acc_norm_stderr"] = df.loc[missing_norm, "acc_stderr"]

    # Final validity mask
    mask = (
        (df["task"] != "all")
        & df["acc_norm"].notna()
        & (df["acc_norm"] > 0)
    )

    return df.loc[mask, ["step", "task", "acc_norm", "acc_norm_stderr"]].copy() 

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across tasks for each *step*."""
    return (
        df.groupby("step").agg(
            mean_acc_norm=("acc_norm", "mean"),
            mean_acc_stderr=("acc_norm_stderr", "mean"),
            n_tasks=("task", "nunique"),
        )
        .reset_index()
    )


def _collect_eval_data(
    results_dir: Path,
    seeds: Sequence[int],
    fname: str,
    *,
    aggregate_tasks: bool,
) -> pd.DataFrame:
    """Collect eval data across seeds, optionally aggregating tasks."""
    parts: list[pd.DataFrame] = []
    for seed in seeds:
        csv_path = results_dir / f"seed={seed}" / fname
        if not csv_path.exists():
            print(f"[eval-warn] {csv_path} not found – skipping seed {seed}")
            continue
        raw = _read_eval_csv(csv_path)
        df_seed = _aggregate(raw) if aggregate_tasks else raw
        df_seed["seed"] = seed
        parts.append(df_seed)

    if not parts:
        raise RuntimeError(f"No eval CSVs under {results_dir} for seeds {list(seeds)}")

    df = pd.concat(parts, ignore_index=True)

    if not aggregate_tasks:
        # combine seeds: mean over seeds for each (step, task)
        df = (
            df.groupby(["step", "task"], as_index=False)
            .agg(acc_norm=("acc_norm", "mean"), acc_norm_stderr=("acc_norm_stderr", "mean"))
        )
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ──────────────────────────────────────────────────────────────────────────────

def plot_lighteval(
    *,
    ax: plt.Axes,
    results_dirs: List[str | Path],
    run_names: List[str],
    styles: List[dict],
    seeds: Sequence[int] | None = None,
    eval_filename: str = "result_lighteval_0shots.csv",
    aggregate_tasks: bool = True,
    task_filter: str | None = None,
    show_markers: bool = True,
    show_errorbars: bool = True,
    marker_size: int = 70,
    remove_x_labels: bool = False,
    remove_y_labels: bool = False,
    remove_legend: bool = False,
    x_scale: str = "linear",
    y_scale: str = "linear",
    subtitle: str | None = None,
    **kwargs,
) -> None:
    """Plot downstream accuracy curves.

    If *task_filter* is provided we plot that single task (one curve per method)
    regardless of *aggregate_tasks*.
    """

    # ── basic checks ────────────────────────────────────────────────────────
    if seeds is None:
        seeds = [666]
    if not (len(results_dirs) == len(run_names) == len(styles)):
        raise ValueError("results_dirs, run_names, styles length mismatch")

    results_dirs = [Path(p) for p in results_dirs]

    palette_cycle = sns.color_palette()

    for idx, (run_name, results_dir, style) in enumerate(zip(run_names, results_dirs, styles)):
        try:
            df = _collect_eval_data(
                results_dir,
                seeds,
                eval_filename,
                aggregate_tasks=False if task_filter else aggregate_tasks,
            )
        except RuntimeError as exc:
            print(f"[warn] {exc}")
            continue

        # Apply task filter if requested ------------------------------------
        if task_filter is not None:
            df = df[df["task"] == task_filter]
            if df.empty:
                print(f"[warn] task '{task_filter}' not found for {results_dir}")
                continue
            # If originally aggregating, aggregate now on filtered task
            if aggregate_tasks:
                df = _aggregate(df)

        # Decide plotting mode ----------------------------------------------
        plot_per_task = not aggregate_tasks and task_filter is None

        base_colour = palette_cycle[idx % len(palette_cycle)]
        colour = style.get("color", base_colour)
        style.setdefault("color", colour)

        if plot_per_task:
            # one curve per task (hue)
            sns.lineplot(
                data=df,
                x="step",
                y="acc_norm",
                hue="task",
                palette="husl",
                ax=ax,
                linewidth=style.get("linewidth", 1.6),
                alpha=style.get("alpha", 0.9),
                legend=not remove_legend,
            )
            if show_errorbars:
                for _, row in df.iterrows():
                    ax.errorbar(row["step"], row["acc_norm"], yerr=row["acc_norm_stderr"], fmt="none", ecolor="grey", capsize=2, lw=1)
            if show_markers:
                ax.scatter(df["step"], df["acc_norm"], s=marker_size*0.6, edgecolor="white", linewidth=0.4)
        else:
            # one curve per method (aggregated or task_filter)
            if aggregate_tasks or task_filter:
                df_plot = df.rename(columns={"mean_acc_norm": "metric", "mean_acc_stderr": "stderr"}) if "mean_acc_norm" in df.columns else df.rename(columns={"acc_norm": "metric", "acc_norm_stderr": "stderr"})
            else:
                raise RuntimeError("Unexpected dataframe format for aggregated plot")

            sns.lineplot(
                data=df_plot,
                x="step",
                y="metric",
                hue=None,
                ax=ax,
                label=run_name,
                errorbar=None,
                **style,
            )
            if show_errorbars:
                ax.errorbar(df_plot["step"], df_plot["metric"], yerr=df_plot["stderr"], fmt="none", ecolor=colour, capsize=3, lw=style.get("linewidth", 2))
            if show_markers:
                ax.scatter(df_plot["step"], df_plot["metric"], s=marker_size, facecolor=colour, edgecolor="white", linewidth=0.6)

    # ── axis / legend housekeeping ─────────────────────────────────────────
    if remove_x_labels:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Steps")

    if remove_y_labels:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Normalized Downstream Acc.")

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    if subtitle is not None:
        ax.set_title(subtitle, fontsize=18)

    if remove_legend:
        leg = ax.get_legend()
        if leg:
            leg.remove()
