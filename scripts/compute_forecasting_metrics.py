"""Compute CRPS and ensemble-mean NRMSE for the CAT probabilistic forecasting
ablation (Appendix `app:forecasting`).

Loads `xs_pred.pt` and `xs_gt.pt` from each of the 8 inference runs under
`logs_paper_forecasting/`, reshapes the saved (T, B*M, C, H, W) tensor into
(T, B, M, C, H, W), strips the C clean-context frames, and computes:

  - CRPS (per pixel, per channel) using the unbiased ensemble form
        crps = mean_i |x_i - y|  -  0.5 * mean_{i,j} |x_i - x_j|
    averaged over pixels (cosine-of-latitude weighted for ERA5) and over the 4
    held-out trajectories, reported per lead time h in {1, 2, 3} and as the mean
    over the three lead times.

  - Ensemble-mean NRMSE: ||mean_i x_i - y||_F / ||y||_F  (or the lat-weighted
    RMSE over the per-channel std for ERA5), averaged over the 4 trajectories
    per lead time and as the mean.

Prints LaTeX-ready table rows for tab:cat_ns, tab:cat_sevir, tab:cat_era5.

Usage::

    python scripts/compute_forecasting_metrics.py
"""
from __future__ import annotations

from pathlib import Path
import json
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO_ROOT / "logs_paper_forecasting"
OUT_DIR = LOG_ROOT / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

M = 16
B = 4
H_LEAD = 3

# ── per-dataset configurations ───────────────────────────────────────────────

NS_RUNS = {
    "rho=0":     "cat_forecast_ns_cr0",
    "cr=0.25":   "cat_forecast_ns_cr0.25",
    "cr=0.75":   "cat_forecast_ns_cr0.75",
}
SEVIR_RUNS = {
    "rho=0":     "cat_forecast_sevir_cr0",
    "cr=0.25":   "cat_forecast_sevir_cr0.25",
    "cr=0.75":   "cat_forecast_sevir_cr0.75",
}
ERA5_RUNS = {
    "rho=0":     "cat_forecast_era5_cr0",
    "cr=0.25":   "cat_forecast_era5_cr0.25",
}

# Context length per dataset (clean context frames at the start of the trajectory)
CTX = {"ns": 10, "sevir": 6, "era5": 6}

VAR_NAMES = ["Z500", "T850", "U10", "V10"]
ERA5_W_NATIVE = 120


def latitude_weights(W: int = ERA5_W_NATIVE) -> torch.Tensor:
    """Cosine-of-latitude area weights, normalised so mean = 1 (matches WB2)."""
    lat_deg = 90.0 - 1.5 * np.arange(W, dtype=np.float64)
    w = np.cos(np.deg2rad(lat_deg)).astype(np.float32)
    w /= w.mean()
    return torch.from_numpy(w)  # (W,)


def _load_run(run_dir: Path):
    """Load xs_pred / xs_gt from either (NS+SEVIR) test_predictions/ or (ERA5) the run dir."""
    cand = [run_dir / "test_predictions" / "xs_pred.pt", run_dir / "xs_pred.pt"]
    for p in cand:
        if p.exists():
            pred = torch.load(p, map_location="cpu", weights_only=False).float()
            gt = torch.load(p.parent / "xs_gt.pt",
                            map_location="cpu", weights_only=False).float()
            return pred, gt
    raise FileNotFoundError(f"No xs_pred.pt under {run_dir}")


def _reshape_to_BM(x: torch.Tensor, B: int = B, M: int = M) -> torch.Tensor:
    """(T, B*M, C, H, W) -> (T, B, M, C, H, W)."""
    T = x.shape[0]
    BM = x.shape[1]
    assert BM == B * M, f"expected B*M={B*M}, got {BM} (shape {tuple(x.shape)})"
    return x.reshape(T, B, M, *x.shape[2:])


def crps_ensemble(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-element unbiased CRPS estimator.

    samples: (..., M, *element_shape)
    truth:   (..., 1, *element_shape) or (..., *element_shape)

    Returns: (..., *element_shape)
    """
    if truth.shape == samples.shape[:-len(samples.shape) + 0]:  # never true; keep simple
        pass
    # Make truth broadcast against the ensemble axis at position -1-len(elem_shape)
    # i.e. truth needs to be (..., 1, *element_shape) so broadcast w/ samples.
    if truth.dim() < samples.dim():
        truth = truth.unsqueeze(-1 * (samples.dim() - truth.dim()))
    M_ = samples.shape[-1 * (samples.dim() - truth.dim() + 1)]  # the M axis size

    # Accuracy term: mean_i |x_i - y|
    diff = (samples - truth).abs()
    M_axis = -1 * (samples.dim() - truth.dim() + 1)
    acc = diff.mean(dim=M_axis)

    # Sharpness term: 0.5 * mean_{i,j} |x_i - x_j|.  Use unbiased estimator by
    # averaging over all (i,j) pairs including the diagonal (which contributes 0).
    # We expand against itself along the next axis up to keep memory manageable
    # by streaming over i.
    sharpness = torch.zeros_like(acc)
    for i in range(M_):
        si = samples.index_select(M_axis, torch.tensor([i], device=samples.device)).squeeze(M_axis)
        # |si - samples| has same shape as samples; mean over M axis
        sharpness += (si.unsqueeze(M_axis) - samples).abs().mean(dim=M_axis)
    sharpness = 0.5 * sharpness / M_

    return acc - sharpness


def crps_ensemble_simple(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-pixel CRPS via the ensemble form, with samples shape (M, ...) and
    truth shape (...). Returns CRPS of same shape as truth.
    """
    M_ = samples.shape[0]
    # accuracy term
    acc = (samples - truth.unsqueeze(0)).abs().mean(dim=0)
    # sharpness term: 0.5 * mean_{i,j} |x_i - x_j|
    # using sum_{i<j} |x_i - x_j| * 2 / M^2  =  mean over all (i,j) pairs
    sharp = torch.zeros_like(truth)
    for i in range(M_):
        sharp = sharp + (samples[i:i+1] - samples).abs().sum(dim=0)
    sharp = 0.5 * sharp / (M_ * M_)
    return acc - sharp


def _strip_context(pred: torch.Tensor, gt: torch.Tensor, ctx: int):
    """Drop the first ctx frames; return only the H_LEAD prediction frames."""
    return pred[ctx:ctx + H_LEAD], gt[ctx:ctx + H_LEAD]


# ── NS / SEVIR scalar metrics (no lat weights) ───────────────────────────────

def compute_scalar_metrics(pred: torch.Tensor, gt: torch.Tensor):
    """pred: (H, B, M, C, h, w); gt same shape (replicated along M).

    Returns dict per lead time:
      crps[h]:  scalar (mean over pixels, channels, B trajectories)
      nrmse[h]: scalar (||pred_mean - gt|| / ||gt|| averaged over B)
    """
    H, B_, M_, C, H_img, W_img = pred.shape
    out = {"crps": [None] * H, "nrmse": [None] * H}

    for h in range(H):
        # CRPS per pixel/channel, then average over (B, C, H, W)
        # pred[h]: (B, M, C, H, W); gt[h]: (B, M, C, H, W) but M-axis is replicated
        crps_pix = crps_ensemble_simple(
            pred[h].permute(1, 0, 2, 3, 4),  # (M, B, C, H, W)
            gt[h, :, 0],  # (B, C, H, W) -- single GT per traj
        )  # -> (B, C, H, W)
        out["crps"][h] = crps_pix.mean().item()

        # Ensemble mean -> NRMSE per trajectory, then mean over B
        pmean = pred[h].mean(dim=1)  # (B, C, H, W)
        gtb = gt[h, :, 0]  # (B, C, H, W)
        diff = (pmean - gtb).reshape(B_, -1)
        gnorm = gtb.reshape(B_, -1).norm(dim=1).clamp(min=1e-12)
        per_traj_nrmse = (diff.norm(dim=1) / gnorm)  # (B,)
        out["nrmse"][h] = per_traj_nrmse.mean().item()

    out["crps_mean"] = float(np.mean(out["crps"]))
    out["nrmse_mean"] = float(np.mean(out["nrmse"]))
    return out


# ── ERA5 lat-weighted, per-variable metrics ──────────────────────────────────

def compute_era5_metrics(pred: torch.Tensor, gt: torch.Tensor):
    """pred: (H, B, M, C=4, lon=240, lat=120); gt same.

    Returns dict[var][metric][h] (lat-weighted CRPS / lat-weighted NRMSE).
    """
    H, B_, M_, C, Hi, Wi = pred.shape
    assert C == 4 and Wi == ERA5_W_NATIVE, f"unexpected ERA5 shape {pred.shape}"
    w = latitude_weights(Wi).view(1, 1, 1, 1, -1)  # broadcast over (B,M,C,H,W)

    out = {var: {"crps": [None] * H, "nrmse": [None] * H} for var in VAR_NAMES}

    for h in range(H):
        # pred[h]: (B, M, C, H, W)
        # CRPS per pixel/channel
        crps_pix = crps_ensemble_simple(
            pred[h].permute(1, 0, 2, 3, 4),  # (M, B, C, H, W)
            gt[h, :, 0],  # (B, C, H, W)
        )  # -> (B, C, H, W)
        # Lat-weighted average over (H, W), then mean over B per variable
        ww = w.squeeze(1).squeeze(0)  # (1, 1, W)
        # crps_pix (B, C, H, W)
        crps_lw = (crps_pix * ww).mean(dim=(-2, -1))  # (B, C)
        for c, var in enumerate(VAR_NAMES):
            out[var]["crps"][h] = crps_lw[:, c].mean().item()

        # Ensemble-mean NRMSE: lat-weighted RMSE per (B, C) of pmean - gt, then mean over B
        pmean = pred[h].mean(dim=1)  # (B, C, H, W)
        gtb = gt[h, :, 0]
        diff2 = (pmean - gtb).pow(2)  # (B, C, H, W)
        rmse_bc = (diff2 * ww).mean(dim=(-2, -1)).sqrt()  # (B, C)
        for c, var in enumerate(VAR_NAMES):
            out[var]["nrmse"][h] = rmse_bc[:, c].mean().item()

    for var in VAR_NAMES:
        out[var]["crps_mean"] = float(np.mean(out[var]["crps"]))
        out[var]["nrmse_mean"] = float(np.mean(out[var]["nrmse"]))
    return out


# ── pretty-printing ──────────────────────────────────────────────────────────

def _fmt(x, n=3):
    return f"{x:.{n}f}"


def print_ns_or_sevir_table(name: str, runs: dict, dataset_key: str):
    print(f"\n{'='*80}\n{name} CRPS / ensemble-mean NRMSE  (per lead time h, mean over 3)\n{'='*80}")
    print(f"{'Method':<22}  {'CRPS h=1':>9} {'CRPS h=2':>9} {'CRPS h=3':>9} {'CRPS mean':>10} | "
          f"{'NRMSE h=1':>10} {'NRMSE h=2':>10} {'NRMSE h=3':>10} {'NRMSE mean':>11}")

    rows = {}
    for label, run in runs.items():
        run_dir = LOG_ROOT / run
        try:
            pred, gt = _load_run(run_dir)
        except FileNotFoundError as e:
            print(f"{label:<22}  MISSING ({run})")
            continue
        pred = _reshape_to_BM(pred)
        gt = _reshape_to_BM(gt)
        # sanity: GT bit-identical along M axis per traj
        gt_max_dev = (gt - gt[:, :, 0:1]).abs().max().item()
        assert gt_max_dev == 0.0, f"{run}: GT not replicated bit-identically (max dev {gt_max_dev})"

        ctx = CTX[dataset_key]
        pred_h, gt_h = _strip_context(pred, gt, ctx)
        m = compute_scalar_metrics(pred_h, gt_h)
        rows[label] = m
        print(f"{label:<22}  {_fmt(m['crps'][0]):>9} {_fmt(m['crps'][1]):>9} {_fmt(m['crps'][2]):>9} {_fmt(m['crps_mean']):>10} | "
              f"{_fmt(m['nrmse'][0]):>10} {_fmt(m['nrmse'][1]):>10} {_fmt(m['nrmse'][2]):>10} {_fmt(m['nrmse_mean']):>11}")

    # LaTeX rows
    print(f"\n--- LaTeX rows for tab:cat_{dataset_key} ---")
    label_map = {
        "rho=0":   "Without CAT ($\\rho = 0$)",
        "cr=0.25": "With CAT ($\\rho = 0.25$)",
        "cr=0.75": "With CAT ($\\rho = 0.75$)",
    }
    for label, m in rows.items():
        cells = [
            _fmt(m["crps"][0]), _fmt(m["crps"][1]), _fmt(m["crps"][2]), _fmt(m["crps_mean"]),
            _fmt(m["nrmse"][0]), _fmt(m["nrmse"][1]), _fmt(m["nrmse"][2]), _fmt(m["nrmse_mean"]),
        ]
        print(f"{label_map[label]:<28} & {' & '.join(cells)} \\\\")

    return rows


def print_era5_table(runs: dict):
    print(f"\n{'='*80}\nERA5 CRPS / ensemble-mean NRMSE per variable  (lat-weighted, per lead time)\n{'='*80}")

    rows = {}
    for label, run in runs.items():
        run_dir = LOG_ROOT / run
        try:
            pred, gt = _load_run(run_dir)
        except FileNotFoundError:
            print(f"{label}: MISSING ({run})")
            continue
        pred = _reshape_to_BM(pred)
        gt = _reshape_to_BM(gt)
        gt_max_dev = (gt - gt[:, :, 0:1]).abs().max().item()
        assert gt_max_dev == 0.0, f"{run}: GT not bit-identical along M axis"

        ctx = CTX["era5"]
        pred_h, gt_h = _strip_context(pred, gt, ctx)
        m = compute_era5_metrics(pred_h, gt_h)
        rows[label] = m

        print(f"\n  {label}:")
        for var in VAR_NAMES:
            d = m[var]
            print(f"    {var:5s}  CRPS  h=1 {_fmt(d['crps'][0])}  h=2 {_fmt(d['crps'][1])}  "
                  f"h=3 {_fmt(d['crps'][2])}  mean {_fmt(d['crps_mean'])}   |   "
                  f"NRMSE h=1 {_fmt(d['nrmse'][0])}  h=2 {_fmt(d['nrmse'][1])}  "
                  f"h=3 {_fmt(d['nrmse'][2])}  mean {_fmt(d['nrmse_mean'])}")

    # LaTeX rows for tab:cat_era5
    print(f"\n--- LaTeX rows for tab:cat_era5 ---")
    label_map = {
        "rho=0":   "Without CAT ($\\rho = 0$)",
        "cr=0.25": "With CAT ($\\rho = 0.25$)",
        "cr=0.75": "With CAT ($\\rho = 0.75$)",
    }
    for label, m in rows.items():
        print(f"\\multirow{{4}}{{*}}{{{label_map[label]}}}")
        for c, var in enumerate(VAR_NAMES):
            d = m[var]
            cells = [
                _fmt(d["crps"][0]), _fmt(d["crps"][1]), _fmt(d["crps"][2]), _fmt(d["crps_mean"]),
                _fmt(d["nrmse"][0]), _fmt(d["nrmse"][1]), _fmt(d["nrmse"][2]), _fmt(d["nrmse_mean"]),
            ]
            print(f" & {var:5s} & {' & '.join(cells)} \\\\")
        print(r"\midrule")
    return rows


def main():
    ns_results    = print_ns_or_sevir_table("NS",    NS_RUNS,    "ns")
    sevir_results = print_ns_or_sevir_table("SEVIR", SEVIR_RUNS, "sevir")
    era5_results  = print_era5_table(ERA5_RUNS)

    out = {
        "ns": ns_results,
        "sevir": sevir_results,
        "era5": era5_results,
    }
    json_path = OUT_DIR / "forecasting_metrics.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {json_path}")


if __name__ == "__main__":
    main()
