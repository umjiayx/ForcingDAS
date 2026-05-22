"""Per-frame relative spectrum-error metric, split into wavenumber bands.

Two operators are provided:

  * `ns_kinetic_energy_spectrum_2d` — for Navier-Stokes vorticity fields
    (matches `spectrum_error_comparison/ns/comparison/compute_spectrum_error.py`
    exactly: vorticity → kinetic-energy spectrum E(k) via stream function).

  * `raw_radial_power_spectrum_2d` — generic 2D radial power spectrum
    (|FFT(field)|^2 binned by |k|), used for non-NS fields such as SEVIR VIL.

The reported per-band metric is the mean per-bin relative error,

    err_band = mean_{k in band} |E_pred(k) - E_gt(k)| / max(E_gt(k), eps)

with `eps = 1e-12 * max_k(E_gt)` per frame. This metric is scale-invariant
(see compute_spectrum_error.py for the design rationale).

Wavenumber bands (for an H=W=128 grid where Nyquist|k|=64):
    low_k  : [0.5, 8)    energy-containing / large scales        ( 8 bins)
    mid_k  : [8,  32)    inertial range                          (24 bins)
    high_k : [32, 64)    near-dissipation / small scales         (32 bins)

Use `compute_spectrum_error_bands(xs_pred, xs, mode, ...)` to get a dict of
`spec_err_low_k / spec_err_mid_k / spec_err_high_k / spec_err_mean` ready to
merge into the lightning test metric_dict.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from numpy.fft import fft2, fftshift


# Default 128x128 bands. Override via the `k_bands` argument if needed.
DEFAULT_K_BANDS: Sequence[tuple[str, float, float]] = (
    ("low_k",  0.5,  8.0),
    ("mid_k",  8.0,  32.0),
    ("high_k", 32.0, 64.0),
)


def ns_kinetic_energy_spectrum_2d(field_2d: np.ndarray, dx: float):
    """Radial kinetic-energy spectrum of a 2D vorticity slice.

    1:1 with FlowDAS_baseline/.../main.py and
    spectrum_error_comparison/ns/comparison/compute_spectrum_error.py.
    """
    vorticity_hat = fft2(field_2d)
    ny, nx = field_2d.shape
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dx) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = np.inf  # kill DC mode

    psi_hat = -vorticity_hat / k2
    u_hat = -1j * ky * psi_hat
    v_hat = 1j * kx * psi_hat
    E_hat = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    k_grid = np.sqrt(kx ** 2 + ky ** 2)
    k_grid = fftshift(k_grid)
    E_hat = fftshift(E_hat)

    k_bins = np.arange(0.5, np.max(k_grid), 1.0)
    energy_spectrum = np.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i + 1])
        energy_spectrum[i] = np.sum(E_hat[mask])
    return k_bins[:-1], energy_spectrum


def raw_radial_power_spectrum_2d(field_2d: np.ndarray, dx: float):
    """Generic 2D radial power spectrum |FFT(field)|^2, binned by |k|.

    Same wavenumber convention as the NS kinetic-energy operator above so
    band cutoffs remain comparable.
    """
    field_hat = fft2(field_2d)
    ny, nx = field_2d.shape
    kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, dx) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    P_hat = np.abs(field_hat) ** 2
    P_hat[0, 0] = 0.0  # zero DC mode (don't blow up the relative error)

    k_grid = np.sqrt(kx ** 2 + ky ** 2)
    k_grid = fftshift(k_grid)
    P_hat = fftshift(P_hat)

    k_bins = np.arange(0.5, np.max(k_grid), 1.0)
    power_spectrum = np.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i + 1])
        power_spectrum[i] = np.sum(P_hat[mask])
    return k_bins[:-1], power_spectrum


_OPERATORS = {
    "kinetic_energy": ns_kinetic_energy_spectrum_2d,
    "raw_power":      raw_radial_power_spectrum_2d,
}


def compute_spectrum_error_bands(
    xs_pred: torch.Tensor,
    xs_gt: torch.Tensor,
    mode: str = "kinetic_energy",
    k_bands: Sequence[tuple[str, float, float]] = DEFAULT_K_BANDS,
    dx: float | None = None,
    eps_rel: float = 1e-12,
) -> dict[str, float]:
    """Mean per-bin relative spectrum error per band, averaged over (T, B).

    Args:
        xs_pred / xs_gt: (T, B, C, H, W) tensors.  C must be 1.  Inputs may be
            in any consistent space — the relative metric is scale-invariant.
        mode: "kinetic_energy" (NS) or "raw_power" (SEVIR / generic).
        k_bands: list of (name, k_lo, k_hi).  Defaults to the 128x128 bands.
        dx: physical grid spacing.  Defaults to ``2*pi / H`` (matches NS).
        eps_rel: per-frame floor scale for E_gt to avoid division blow-up.

    Returns:
        dict with keys ``spec_err_<band>`` for each band plus ``spec_err_mean``
        (average across bands).  All values are Python floats.
    """
    assert xs_pred.shape == xs_gt.shape, (xs_pred.shape, xs_gt.shape)
    assert xs_pred.dim() == 5 and xs_pred.shape[2] == 1, (
        "expected (T, B, 1, H, W); got " + str(tuple(xs_pred.shape))
    )
    if mode not in _OPERATORS:
        raise ValueError(f"unknown spectrum mode {mode!r}; expected one of {list(_OPERATORS)}")
    spectrum_op = _OPERATORS[mode]

    pred_np = xs_pred.detach().cpu().float().numpy()[:, :, 0]  # (T, B, H, W)
    gt_np = xs_gt.detach().cpu().float().numpy()[:, :, 0]
    T, B, H, _ = pred_np.shape
    if dx is None:
        dx = 2 * np.pi / H

    band_errs: dict[str, list[float]] = {name: [] for name, _, _ in k_bands}
    for t in range(T):
        for b in range(B):
            try:
                k_bins, E_p = spectrum_op(pred_np[t, b], dx)
                _,      E_g = spectrum_op(gt_np[t, b],   dx)
                eps = eps_rel * max(float(E_g.max()), 1e-30)
                rel = np.abs(E_p - E_g) / np.maximum(E_g, eps)
                for name, k_lo, k_hi in k_bands:
                    mask = (k_bins >= k_lo) & (k_bins < k_hi)
                    if mask.any():
                        band_errs[name].append(float(rel[mask].mean()))
            except Exception:
                # Numerical issues on a single frame shouldn't bring the run down.
                continue

    out: dict[str, float] = {}
    band_means: list[float] = []
    for name, _, _ in k_bands:
        v = float(np.mean(band_errs[name])) if band_errs[name] else float("nan")
        out[f"spec_err_{name}"] = v
        if np.isfinite(v):
            band_means.append(v)
    out["spec_err_mean"] = float(np.mean(band_means)) if band_means else float("nan")
    return out
