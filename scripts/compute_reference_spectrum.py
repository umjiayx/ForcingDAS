#!/usr/bin/env python3
"""
Pre-compute radially-averaged reference power spectra for NS and SEVIR.

NS:    raw vorticity data converted to minmax [0,1] space.
SEVIR: data already in [0,1].

Outputs saved to ckpts/ alongside model checkpoints.

Usage:
    python scripts/compute_reference_spectrum.py
"""

import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "algorithms" / "diffusion_forcing"))
from spectral_utils import build_radial_bin_index


def _chunked_reference_spectrum(data_path, bin_index, n_bins, preprocess=None,
                                chunk_trajs=50):
    """
    Compute reference spectrum in chunks to avoid OOM.
    Loads the full file via mmap, processes `chunk_trajs` trajectories at a time.
    """
    data = torch.load(data_path, map_location="cpu", mmap=True)
    N = data.shape[0]
    print(f"  shape={tuple(data.shape)}, processing in chunks of {chunk_trajs} trajs")

    flat_idx = bin_index.reshape(-1)
    accum = torch.zeros(n_bins, dtype=torch.float64)
    count_per_bin = torch.zeros(n_bins, dtype=torch.float64)
    count_per_bin.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float64))
    count_per_bin = count_per_bin.clamp(min=1)

    n_samples = 0
    for start in range(0, N, chunk_trajs):
        end = min(start + chunk_trajs, N)
        chunk = data[start:end].float()
        if preprocess is not None:
            chunk = preprocess(chunk)
        x = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])
        n_samples += x.shape[0]

        fft_x = torch.fft.rfft2(x)
        power = fft_x.abs().pow(2)
        flat_power = power.reshape(-1, power.shape[-2] * power.shape[-1])

        idx = flat_idx.unsqueeze(0).expand(flat_power.shape[0], -1)
        chunk_sum = torch.zeros(flat_power.shape[0], n_bins, dtype=torch.float64)
        chunk_sum.scatter_add_(1, idx, flat_power.double())
        bin_mean_per_sample = chunk_sum / count_per_bin.unsqueeze(0)
        accum += bin_mean_per_sample.sum(dim=0)

        print(f"    processed trajs [{start}:{end}), total frames so far: {n_samples}")

    ref_spectrum = (accum / n_samples).float()
    return ref_spectrum


def main():
    out_dir = _root / "ckpts"
    out_dir.mkdir(exist_ok=True)

    H, W = 128, 128
    bin_index, n_bins = build_radial_bin_index(H, W)
    print(f"Radial bins: {n_bins}  (H={H}, W={W})")

    # --- NS (minmax normalization → [0, 1]) ---
    ns_path = "/path/to/ns/train/data.pt"
    data_min, data_max = -19.16, 17.42
    print(f"\nNS: {ns_path}")
    ns_spec = _chunked_reference_spectrum(
        ns_path, bin_index, n_bins,
        preprocess=lambda x: (x - data_min) / (data_max - data_min),
        chunk_trajs=100,
    )
    ns_out = out_dir / "ns_reference_spectrum.pt"
    torch.save(ns_spec, ns_out)
    print(f"  Saved → {ns_out}  shape={tuple(ns_spec.shape)}")

    # --- SEVIR (already in [0, 1]) ---
    sevir_path = "/path/to/sevir/train/data.npy"
    print(f"\nSEVIR: {sevir_path}")
    sevir_spec = _chunked_reference_spectrum(
        sevir_path, bin_index, n_bins,
        chunk_trajs=200,
    )
    sevir_out = out_dir / "sevir_reference_spectrum.pt"
    torch.save(sevir_spec, sevir_out)
    print(f"  Saved → {sevir_out}  shape={tuple(sevir_spec.shape)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
