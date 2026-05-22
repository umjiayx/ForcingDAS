"""
Differentiable radially-averaged power spectrum utilities for spectral loss.

Used by _make_measurement_fn_v4 in df_ns.py and df_sevir.py to regularize
the energy spectrum of x̂_0 during observation guidance guidance.
"""

import torch


def build_radial_bin_index(H: int, W: int):
    """
    Map each rfft2 output cell to a radial wavenumber bin.

    Returns:
        bin_index: LongTensor (H, W//2+1)
        n_bins:    int — number of radial bins (= min(H,W) // 2)
    """
    freq_y = torch.fft.fftfreq(H, d=1.0)
    freq_x = torch.fft.rfftfreq(W, d=1.0)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    k = torch.sqrt(fy ** 2 + fx ** 2)

    n_bins = min(H, W) // 2
    k_scaled = k * max(H, W)
    bin_index = k_scaled.long().clamp(max=n_bins - 1)
    return bin_index, n_bins


def radial_power_spectrum(x, bin_index, n_bins):
    """
    Differentiable radially-averaged power spectrum.

    Args:
        x:         (..., H, W) tensor in data space.
        bin_index: (H, W//2+1) LongTensor from build_radial_bin_index.
        n_bins:    int.

    Returns:
        spectrum: (n_bins,) — mean power per radial bin, averaged over all
                  leading dimensions (frames, batch, channels).
    """
    fft_x = torch.fft.rfft2(x)
    power = fft_x.abs().pow(2)

    flat_power = power.reshape(-1, power.shape[-2] * power.shape[-1])
    flat_idx = bin_index.reshape(-1).to(x.device)

    idx = flat_idx.unsqueeze(0).expand(flat_power.shape[0], -1)

    spectrum_sum = torch.zeros(
        flat_power.shape[0], n_bins, device=x.device, dtype=x.dtype
    )
    spectrum_sum.scatter_add_(1, idx, flat_power)

    count = torch.zeros(n_bins, device=x.device, dtype=x.dtype)
    count.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=x.dtype))
    count = count.clamp(min=1)

    spectrum = spectrum_sum / count.unsqueeze(0)
    spectrum = spectrum.mean(dim=0)
    return spectrum


def compute_frequency_weight(alpha, n_bins, sharpness=5.0):
    """
    Diffusion-aware soft mask: progressively includes higher-frequency bins
    as denoising advances (alpha grows toward 1).

    Args:
        alpha:     scalar or (frames,) tensor — cumulative signal fraction.
        n_bins:    int — number of frequency bins.
        sharpness: float — controls sigmoid steepness (default 5.0).

    Returns:
        freq_weight: (n_bins,) tensor on the same device as alpha.
    """
    progress = alpha.mean().clamp(0, 1)
    k_indices = torch.arange(n_bins, device=alpha.device, dtype=alpha.dtype)
    k_cutoff = n_bins * progress
    return torch.sigmoid(sharpness * (k_cutoff - k_indices))


def compute_reference_spectrum(data, bin_index, n_bins):
    """
    Average radial power spectrum over a dataset (non-differentiable).

    Args:
        data:      (N, T, H, W) or (N, H, W) training data in [0,1] space.
        bin_index: (H, W//2+1) LongTensor.
        n_bins:    int.

    Returns:
        ref_spectrum: (n_bins,) tensor.
    """
    with torch.no_grad():
        x = data.reshape(-1, data.shape[-2], data.shape[-1])
        return radial_power_spectrum(x, bin_index, n_bins)
