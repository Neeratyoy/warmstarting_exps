import argparse
import lightning as L
import math
import pandas as pd
from pathlib import Path
import torch
from typing import Dict, Optional

from saws.checkpointer import CheckpointManager


ENDPOINT_FILE = "tb_logs.parquet"
MAX_DEPTH = 10

METRIC_COLUMNS = [
    "l1_norm",
    "l2_norm",
    "rms_norm",
    "effective_rank",
    "op_norm_rms_inf",
    "cosine_to_prev",
]


@torch.no_grad()
def _effective_rank(sigma: torch.Tensor, eps: float = 1e-12) -> float:
    """Roy & Vetterli (2007): erank = exp(H(p)), p = sigma / ||sigma||_1."""
    sigma = sigma[sigma > eps]
    if sigma.numel() == 0:
        return float("nan")
    p = sigma / sigma.sum()
    # entropy in nats; clamp avoids log(0)
    H = -(p * p.clamp_min(eps).log()).sum()
    return float(torch.exp(H).item())


@torch.no_grad()
def _op_norm_rms_to_inf(W: torch.Tensor) -> float:
    """
    ||W||_{RMS -> inf} for W in R^{d_out x d_in}:
        = sup_{||x||_RMS = 1} ||W x||_inf
        = sqrt(d_in) * max_i ||W[i, :]||_2

    (Pethick et al. 2025 / Filatov et al. 2025.)
    """
    if W.ndim != 2:
        return float("nan")
    d_in = W.shape[1]
    row_l2_max = torch.linalg.vector_norm(W, ord=2, dim=1).max()
    return float((math.sqrt(d_in) * row_l2_max).item())


def radial_angular_decomp(W_t: torch.Tensor, W_0: torch.Tensor | None) -> Optional[Dict[str, float]]:
    if W_0 is None:
        return None
    dW = (W_t - W_0).flatten()
    w0 = W_0.flatten()
    w0_norm = torch.linalg.vector_norm(w0)
    if w0_norm == 0:
        return None
    e0 = w0 / w0_norm
    radial = torch.dot(dW, e0)
    angular_vec = dW - radial * e0
    angular_norm = torch.linalg.vector_norm(angular_vec)
    dW_norm = torch.linalg.vector_norm(dW)
    return {
        "radial": float(radial),
        "angular_norm": float(angular_norm),
        "dW_norm": float(dW_norm),
        "radial_fraction": float(radial.abs() / dW_norm) if dW_norm > 0 else 0.0,
        "angular_fraction": float(angular_norm / dW_norm) if dW_norm > 0 else 0.0,
    }


@torch.no_grad()
def compute_tensor_metrics(
    v: torch.Tensor,
    v_prev_flat: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {c: float("nan") for c in METRIC_COLUMNS}

    if v.ndim not in (1, 2):
        return out

    W = v.detach().to(torch.float32)
    n = W.numel()

    out["l1_norm"]  = float(torch.linalg.vector_norm(W, ord=1).item())
    out["l2_norm"]  = float(torch.linalg.vector_norm(W, ord=2).item())
    out["rms_norm"] = out["l2_norm"] / math.sqrt(n)

    # Reshape 1-D to (1, n) so erank / op-norm are defined uniformly.
    W2d = W if W.ndim == 2 else W.unsqueeze(0)

    sigma = torch.linalg.svdvals(W2d)
    out["effective_rank"]  = _effective_rank(sigma)
    out["op_norm_rms_inf"] = _op_norm_rms_to_inf(W2d)

    if v_prev_flat is not None:
        cur_flat = W.flatten()
        denom = torch.linalg.vector_norm(cur_flat) * torch.linalg.vector_norm(v_prev_flat)
        if denom > 0:
            out["cosine_to_prev"] = float(
                torch.dot(cur_flat, v_prev_flat.to(cur_flat.dtype)) / denom
            )

    return out


def recurse_collect_run_paths(base_path: Path, max_depth: int, current_depth: int = 0):
    if current_depth > max_depth:
        return []

    if ENDPOINT_FILE in [child.name for child in base_path.iterdir() if child.is_file()]:
        return [base_path]

    run_paths = []
    for child in base_path.iterdir():
        if child.is_dir():
            if (child / ENDPOINT_FILE).exists():
                run_paths.append(child)
            else:
                run_paths.extend(recurse_collect_run_paths(child, max_depth, current_depth + 1))
    
    return run_paths


def get_args():
    parser = argparse.ArgumentParser(description="Parser for generating MuP base files")
    parser.add_argument(
        "--base_path",
        type=str,
        help="The path to the base canvas file",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=MAX_DEPTH,
        help="The maximum depth to traverse the base_path to collect run paths",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    fabric = L.Fabric(accelerator="auto", devices="auto")

    run_paths = recurse_collect_run_paths(Path(args.base_path), args.max_depth)

    for _path in run_paths:
        print(f"| Processing run path: {_path}")
        
        ckpt_mgr = CheckpointManager(
            load_dir=_path,
            fabric=fabric,
        )
        ckpts = ckpt_mgr._list_available_checkpoints()
        ids = sorted(list(ckpts.keys()))
        if -1 in ids:
            ids.remove(-1)


        # in main, replacing the inner for-loop:
        records = []
        prev_flats: Dict[str, torch.Tensor] = {}
        init_flats: Dict[str, torch.Tensor] = {}

        for _id in ids:
            print(f"|-- Processing checkpoint: {ckpts[_id].name}")
            remainder, _ = ckpt_mgr.load_checkpoint(state=None, train_step=_id)
            model = remainder["model"]
            del remainder

            n_keys = len(model)
            for i, (k, v) in enumerate(model.items(), start=1):
                print(f"|---- Processing model key {i:2d}/{n_keys:2d}", end="\r")

                if v.ndim not in (1, 2):
                    continue

                if _id == 0:
                    init_flats[k] = v.detach().to(torch.float32).flatten().cpu()

                prev = prev_flats.get(k)
                try:
                    metrics = compute_tensor_metrics(v, v_prev_flat=prev)
                    _add_metrics = radial_angular_decomp(v.flatten(), init_flats.get(k, None))
                except Exception as e:
                    metrics = {}
                    print(f"Error computing metrics for key {k} at step {_id}: {e}")
                    continue
                
                metrics.update(
                    {
                        "radial": float("nan"),
                        "angular_norm": float("nan"),
                        "dW_norm": float("nan"),
                        "radial_fraction": float("nan"),
                        "angular_fraction": float("nan"),
                    } if _add_metrics is None else _add_metrics
                )
                records.append({"key": k, "step": _id, **metrics})

                # Stash flat tensor on CPU for the next step's cosine.
                prev_flats[k] = v.detach().to(torch.float32).flatten().cpu()
            print()  # newline after the \r progress

        # Build the dataframe for this run path.
        df = (
            pd.DataFrame.from_records(records)
            .set_index(["key", "step"])
            .sort_index()
        )
        ### to read an entry: df.loc[("model.layer1.weight", 100)]  # example for key="model.layer1.weight" and step=1000

        out_path = _path / "post_hoc_analysis.parquet"
        df.to_parquet(out_path)
        print(f"| Wrote {out_path}  ({len(df)} rows)")

# end of file