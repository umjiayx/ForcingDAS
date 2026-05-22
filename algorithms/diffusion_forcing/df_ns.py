"""
Diffusion Forcing for Navier-Stokes vorticity data (main branch / 3D U-Net).

Extends DiffusionForcingVideo with:
  - icefire colormap visualization (1-ch vorticity → 3-ch RGB)
  - Normalization-aware visualization and metrics
  - Two normalization modes:
      "minmax" — min-max to [0,1], then _normalize_x → [-1,1]
      "zscore" — raw / data_raw_std → mean≈0, std≈1, range≈[-6,6]
  - observation guidance for inverse problems at test time
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.fft import fft2, fftshift
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
import wandb

from .df_video import DiffusionForcingVideo
from .df_base import DiffusionForcingBase
from .operators import get_operator
from .spectral_utils import build_radial_bin_index, radial_power_spectrum, compute_frequency_weight
from .spectrum_error_utils import compute_spectrum_error_bands
from utils.logging_utils import log_video, get_validation_metrics_for_videos

try:
    import seaborn as _sns
    _VORTICITY_CMAP = _sns.cm.icefire
except (ImportError, AttributeError):
    _VORTICITY_CMAP = plt.colormaps.get_cmap("coolwarm")


class DiffusionForcingNS(DiffusionForcingVideo):
    """
    DiffusionForcingVideo subclass for 1-channel Navier-Stokes vorticity.

    1. Normalization mode ("minmax" or "zscore") determines how data maps
       to/from training space, how metrics are computed, and how vis works.
    2. Overrides training_step and on_validation_epoch_end so wandb videos
       use the icefire colormap instead of raw single-channel data.
    3. For "zscore" mode, clip_noise is raised to 15 to avoid clamping
       extreme data values during generation.
    """

    def __init__(self, cfg: DictConfig):
        self.normalization = cfg.normalization
        self.data_min = cfg.data_min
        self.data_max = cfg.data_max
        self.data_raw_std = cfg.data_raw_std
        self.vis_vmin = cfg.vis_vmin
        self.vis_vmax = cfg.vis_vmax
        self.train_vis_every = cfg.train_vis_every

        if self.normalization == "zscore":
            self._metrics_data_range = (self.data_max - self.data_min) / self.data_raw_std
            cfg.data_mean = 0.0
            cfg.data_std = 1.0
            cfg.diffusion.clip_noise = 15.0
        else:
            self._metrics_data_range = 1.0
            cfg.data_mean = 0.5
            cfg.data_std = 0.5

        obs_cfg = cfg.get("obs_guidance", None)
        self.obs_enabled = obs_cfg is not None and obs_cfg.get("enabled", False)
        if self.obs_enabled:
            self.obs_grad_scale = float(obs_cfg.grad_scale)
            self.obs_noise_sigma = float(obs_cfg.noise_sigma)
            self.obs_gamma = float(obs_cfg.get("gamma", 0.0))
            op_cfg = obs_cfg.operator
            self.obs_operator = get_operator(op_cfg)
            print(f"[obs-guidance] Enabled: operator={self.obs_operator}, "
                  f"noise_sigma={self.obs_noise_sigma}, grad_scale={self.obs_grad_scale}, "
                  f"gamma={self.obs_gamma}")

            self.obs_spectral_lambda = float(obs_cfg.get("spectral_lambda", 0.0))
            self.obs_spectral_sharpness = float(obs_cfg.get("spectral_sharpness", 5.0))
            spec_ref_path = obs_cfg.get("spectral_ref", "")
            if self.obs_spectral_lambda > 0 and spec_ref_path:
                self._spectral_bin_index, self._spectral_n_bins = build_radial_bin_index(128, 128)
                self._spectral_ref = torch.load(spec_ref_path, map_location="cpu")
                print(f"[obs-guidance] Spectral reg: lambda={self.obs_spectral_lambda}, "
                      f"sharpness={self.obs_spectral_sharpness}, "
                      f"ref={spec_ref_path} ({self._spectral_n_bins} bins)")
            else:
                self.obs_spectral_lambda = 0.0

        super().__init__(cfg)

    def _vorticity_to_rgb(self, x):
        """
        Convert 1-channel vorticity tensor to 3-channel RGB via colormap.

        minmax: x in [0,1] after _unnormalize_x → reverse to original → divide by std → colormap
        zscore: x already in unit-std space → colormap directly

        Args:
            x: (frame, batch, 1, H, W) after _unnormalize_x.
        Returns:
            (frame, batch, 3, H, W) in [0, 1].
        """
        v = x.detach().cpu().numpy()

        if self.normalization == "zscore":
            pass
        else:
            v = v * (self.data_max - self.data_min) + self.data_min
            v = v / self.data_raw_std

        v = v.squeeze(2)  # (F, B, H, W)

        norm = mcolors.Normalize(vmin=self.vis_vmin, vmax=self.vis_vmax, clip=True)
        rgba = _VORTICITY_CMAP(norm(v))  # (F, B, H, W, 4)
        rgb = rgba[..., :3]
        rgb = np.transpose(rgb, (0, 1, 4, 2, 3))  # (F, B, 3, H, W)

        return torch.from_numpy(rgb.astype(np.float32))

    @staticmethod
    def _energy_spectrum(vorticity_2d):
        """Kinetic energy spectrum from a 2D vorticity field (H, W)."""
        ny, nx = vorticity_2d.shape
        dx = 2.0 * np.pi / nx
        vort_hat = fft2(vorticity_2d)
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dx) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)
        k2 = kx ** 2 + ky ** 2
        k2[0, 0] = np.inf
        psi_hat = -vort_hat / k2
        u_hat = -1j * ky * psi_hat
        v_hat = 1j * kx * psi_hat
        E_hat = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
        k_mag = fftshift(np.sqrt(kx ** 2 + ky ** 2))
        E_hat = fftshift(E_hat)
        k_bins = np.arange(0.5, np.max(k_mag), 1.0)
        spectrum = np.zeros(len(k_bins) - 1)
        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            spectrum[i] = np.sum(E_hat[mask])
        return k_bins[:-1], spectrum

    def _log_frame_grid(self, pred_rgb, gt_rgb, pred_raw, gt_raw, logger,
                        namespace, step=None, frame_stride=2, sample_idx=0,
                        title_prefix=""):
        """Log a 3-row (GT / Pred / Energy Spectrum) x N-column grid to wandb."""
        n_frames = gt_rgb.shape[0]
        frame_indices = list(range(0, n_frames, frame_stride))
        if (n_frames - 1) not in frame_indices:
            frame_indices.append(n_frames - 1)
        n = len(frame_indices)

        fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8))
        if n == 1:
            axes = axes[:, None]

        for col, f in enumerate(frame_indices):
            gt_img = gt_rgb[f, sample_idx].permute(1, 2, 0).numpy()
            pred_img = pred_rgb[f, sample_idx].permute(1, 2, 0).numpy()

            axes[0, col].imshow(np.clip(gt_img, 0, 1))
            axes[0, col].set_title(f"t={f}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(np.clip(pred_img, 0, 1))
            axes[1, col].axis("off")

            gt_vort = gt_raw[f, sample_idx, 0].numpy()
            pred_vort = pred_raw[f, sample_idx, 0].numpy()
            k_gt, spec_gt = self._energy_spectrum(gt_vort)
            k_pred, spec_pred = self._energy_spectrum(pred_vort)
            axes[2, col].loglog(k_gt, spec_gt, "b-", linewidth=1, label="GT")
            axes[2, col].loglog(k_pred, spec_pred, "r--", linewidth=1, label="Pred")
            axes[2, col].set_xticks([])
            axes[2, col].set_yticks([])
            if col == 0:
                axes[2, col].legend(fontsize=7, loc="upper right")

        axes[0, 0].set_ylabel("GT", fontsize=12, rotation=0, labelpad=30, va="center")
        axes[1, 0].set_ylabel("Pred", fontsize=12, rotation=0, labelpad=30, va="center")
        axes[2, 0].set_ylabel("E(k)", fontsize=12, rotation=0, labelpad=30, va="center")

        sup_title = f"{title_prefix}Frames Comparison" if title_prefix else "Frames Comparison"
        fig.suptitle(sup_title, fontsize=13, y=1.0)
        fig.tight_layout()

        log_key = f"{namespace}/frame_grid"
        if step is not None:
            log_key = f"{log_key}_{step}"
        logger.log({log_key: wandb.Image(fig)})
        plt.close(fig)

    def _make_measurement_fn(self, y_obs, from_noise_levels, to_noise_levels):
        """
        Build a differentiable measurement loss for one DDIM step.

        Observations y_obs live in the same space as _unnormalize_x output:
          - minmax: [0, 1]
          - zscore: raw / data_raw_std  (mean ≈ 0, std ≈ 1)
        """
        active = (from_noise_levels > to_noise_levels)
        sigma_y = self.obs_noise_sigma
        operator = self.obs_operator

        def measurement_fn(x_hat_0):
            x_data = self._unnormalize_x(x_hat_0)
            if self.normalization == "minmax":
                x_data = x_data.clamp(0, 1)
            shape = x_data.shape
            x_flat = x_data.reshape(-1, *shape[-3:])
            Ax_flat = operator(x_flat)
            Ax = Ax_flat.reshape(*shape[:2], *Ax_flat.shape[-3:])

            residuals = y_obs - Ax
            mask = active.float()
            while mask.ndim < residuals.ndim:
                mask = mask.unsqueeze(-1)
            residuals = residuals * mask
            norm = torch.linalg.norm(residuals)
            if sigma_y > 0:
                norm = norm / sigma_y
            return norm

        return measurement_fn

    def _make_measurement_fn_v3(self, y_obs, from_noise_levels, to_noise_levels):
        """
        Build a differentiable measurement loss with SDA variance reweighting.

        Per-frame residuals are divided by sqrt(sigma_y^2 + gamma*(1-alpha)/alpha)
        instead of a flat sigma_y. This suppresses guidance for noisy frames whose
        x̂_0 prediction is unreliable.

        When gamma=0 this reduces to v1 behavior (flat 1/sigma_y weighting).
        """
        active = (from_noise_levels > to_noise_levels)
        sigma_y = self.obs_noise_sigma
        gamma = self.obs_gamma
        operator = self.obs_operator

        dm = self.diffusion_model
        real_steps = torch.linspace(
            -1, dm.timesteps - 1,
            steps=dm.sampling_timesteps + 1,
            device=from_noise_levels.device,
        ).long()
        real_t = real_steps[from_noise_levels]
        real_t_clipped = real_t.clamp(min=0)
        alpha = dm.alphas_cumprod[real_t_clipped]

        alpha_safe = alpha.clamp(min=1e-6)
        per_frame_var = sigma_y ** 2 + gamma * (1.0 - alpha_safe) / alpha_safe
        per_frame_inv_std = 1.0 / per_frame_var.sqrt()

        def measurement_fn(x_hat_0):
            x_data = self._unnormalize_x(x_hat_0)
            if self.normalization == "minmax":
                x_data = x_data.clamp(0, 1)
            shape = x_data.shape
            x_flat = x_data.reshape(-1, *shape[-3:])
            Ax_flat = operator(x_flat)
            Ax = Ax_flat.reshape(*shape[:2], *Ax_flat.shape[-3:])

            residuals = y_obs - Ax

            mask = active.float()
            while mask.ndim < residuals.ndim:
                mask = mask.unsqueeze(-1)
            residuals = residuals * mask

            w = per_frame_inv_std.clone()
            while w.ndim < residuals.ndim:
                w = w.unsqueeze(-1)
            residuals = residuals * w

            return torch.linalg.norm(residuals)

        return measurement_fn

    def _make_measurement_fn_v4(self, y_obs, from_noise_levels, to_noise_levels):
        """
        SDA-reweighted pixel loss + diffusion-aware spectral regularization.

        Combines v3 pixel loss with a log-space spectral penalty that uses a
        progressive frequency mask: low frequencies are constrained early in
        denoising, high frequencies only later.
        """
        active = (from_noise_levels > to_noise_levels)
        sigma_y = self.obs_noise_sigma
        gamma = self.obs_gamma
        operator = self.obs_operator
        spectral_lambda = self.obs_spectral_lambda
        spectral_ref = self._spectral_ref.to(from_noise_levels.device)
        bin_index = self._spectral_bin_index
        n_bins = self._spectral_n_bins
        sharpness = self.obs_spectral_sharpness

        dm = self.diffusion_model
        real_steps = torch.linspace(
            -1, dm.timesteps - 1,
            steps=dm.sampling_timesteps + 1,
            device=from_noise_levels.device,
        ).long()
        real_t = real_steps[from_noise_levels]
        real_t_clipped = real_t.clamp(min=0)
        alpha = dm.alphas_cumprod[real_t_clipped]

        alpha_safe = alpha.clamp(min=1e-6)
        per_frame_var = sigma_y ** 2 + gamma * (1.0 - alpha_safe) / alpha_safe
        per_frame_inv_std = 1.0 / per_frame_var.sqrt()

        freq_weight = compute_frequency_weight(alpha_safe.flatten(), n_bins, sharpness)

        def measurement_fn(x_hat_0):
            x_data = self._unnormalize_x(x_hat_0)
            if self.normalization == "minmax":
                x_data = x_data.clamp(0, 1)
            shape = x_data.shape
            x_flat = x_data.reshape(-1, *shape[-3:])
            Ax_flat = operator(x_flat)
            Ax = Ax_flat.reshape(*shape[:2], *Ax_flat.shape[-3:])

            residuals = y_obs - Ax
            mask = active.float()
            while mask.ndim < residuals.ndim:
                mask = mask.unsqueeze(-1)
            residuals = residuals * mask

            w = per_frame_inv_std.clone()
            while w.ndim < residuals.ndim:
                w = w.unsqueeze(-1)
            residuals = residuals * w
            pixel_loss = torch.linalg.norm(residuals)

            pred_spec = radial_power_spectrum(x_data, bin_index, n_bins)
            eps = 1e-8
            ref = spectral_ref.clone()
            fw = freq_weight.clone()
            log_ratio = torch.log(pred_spec + eps) - torch.log(ref + eps)
            spectral_loss = (fw * log_ratio ** 2).mean()

            return pixel_loss + spectral_lambda * spectral_loss

        return measurement_fn

    @torch.no_grad()
    def _obs_validation_step(self, batch, batch_idx, namespace="test"):
        """Sampling loop with observation guidance guidance — replaces base validation_step."""
        xs, conditions, masks = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape

        xs_unnorm = self._unnormalize_x(xs)
        if self.normalization == "minmax":
            xs_unnorm = xs_unnorm.clamp(0, 1)
        xs_flat = xs_unnorm.reshape(-1, *xs_unnorm.shape[-3:])
        y_flat = self.obs_operator(xs_flat)
        y_clean = y_flat.reshape(*xs_unnorm.shape[:2], *y_flat.shape[-3:])
        y_obs = y_clean + self.obs_noise_sigma * torch.randn_like(y_clean)

        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame = n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="observation guidance Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1])
                )[:, None].repeat(batch_size, axis=1)

                from_nl = torch.from_numpy(from_noise_levels).to(self.device)
                to_nl = torch.from_numpy(to_noise_levels).to(self.device)

                y_window = y_obs[start_frame : curr_frame + horizon]
                if getattr(self, 'obs_spectral_lambda', 0) > 0:
                    make_fn = self._make_measurement_fn_v4
                elif getattr(self, 'obs_gamma', 0) > 0:
                    make_fn = self._make_measurement_fn_v3
                else:
                    make_fn = self._make_measurement_fn
                meas_fn = make_fn(
                    y_window, from_nl[start_frame:], to_nl[start_frame:]
                )

                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon],
                    from_nl[start_frame:],
                    to_nl[start_frame:],
                    measurement_fn=meas_fn,
                    grad_scale=self.obs_grad_scale,
                )

            curr_frame += horizon
            pbar.update(horizon)

        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)

        xs_out = self._unstack_and_unnormalize(xs)
        xs_pred_out = self._unstack_and_unnormalize(xs_pred)
        self.validation_step_outputs.append((xs_pred_out.detach().cpu(), xs_out.detach().cpu()))

        if not hasattr(self, "_obs_observations"):
            self._obs_observations = []
        self._obs_observations.append(y_obs.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        output_dict = DiffusionForcingBase.training_step(self, batch, batch_idx)

        if self.global_step > 0 and self.global_step % self.train_vis_every == 0:
            try:
                _xp = output_dict["xs_pred"]
                _xg = output_dict["xs"]
                print(f"[NS-debug] train xs_pred  range [{_xp.min().item():.4f}, {_xp.max().item():.4f}]  mean={_xp.mean().item():.4f}")
                print(f"[NS-debug] train xs (GT)  range [{_xg.min().item():.4f}, {_xg.max().item():.4f}]  mean={_xg.mean().item():.4f}")

                xs_pred_rgb = self._vorticity_to_rgb(output_dict["xs_pred"])
                xs_rgb = self._vorticity_to_rgb(output_dict["xs"])
                log_video(
                    xs_pred_rgb,
                    xs_rgb,
                    step=self.global_step,
                    namespace="training_vis",
                    logger=self.logger.experiment,
                )
            except Exception as e:
                if not getattr(self, "_video_log_warned", False):
                    print(f"[DiffusionForcingNS] Warning: training video logging failed: {e}")
                    self._video_log_warned = True

        return output_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        if self.obs_enabled:
            loss = self._obs_validation_step(batch, batch_idx, namespace="test")
        else:
            loss = self.validation_step(batch, batch_idx, *args, **kwargs, namespace="test")

        pred, gt = self.validation_step_outputs[-1]
        try:
            pred_rgb = self._vorticity_to_rgb(pred)
            gt_rgb = self._vorticity_to_rgb(gt)
            bi = len(self.validation_step_outputs) - 1
            log_video(
                pred_rgb,
                gt_rgb,
                step=None,
                namespace=f"test_vis/batch_{bi}",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )
            n_samples = pred.shape[1]
            for s in range(n_samples):
                self._log_frame_grid(
                    pred_rgb, gt_rgb, pred, gt,
                    logger=self.logger.experiment,
                    namespace=f"test_frames/batch_{bi}",
                    step=s,
                    sample_idx=s,
                    title_prefix=f"Batch {bi} Sample {s} — ",
                )
            if self.obs_enabled and hasattr(self, "_obs_observations"):
                self._log_obs_observation(
                    self._obs_observations[-1], gt,
                    logger=self.logger.experiment,
                    batch_idx=bi,
                )

            if self.obs_enabled:
                for s in range(n_samples):
                    self._log_nrmse_curve(
                        pred, gt,
                        logger=self.logger.experiment,
                        batch_idx=bi,
                        sample_idx=s,
                    )

            print(f"[DiffusionForcingNS] Logged test batch {bi} video + grid to wandb")
        except Exception as e:
            print(f"[DiffusionForcingNS] Warning: per-batch test vis failed: {e}")

        return loss

    def _log_obs_observation(self, y_obs, gt, logger, batch_idx):
        """Log GT vs observation visualization with energy spectrum."""
        from .operators import SuperResolution, SparseObservation

        gt_np = gt.cpu().numpy()
        y_np = y_obs.cpu().numpy()

        n_frames = gt_np.shape[0]
        frame_indices = list(range(0, n_frames, max(1, n_frames // 8)))
        if (n_frames - 1) not in frame_indices:
            frame_indices.append(n_frames - 1)
        n = len(frame_indices)
        s = 0

        def to_vis(v):
            """Map unnormalized data to vorticity visualization space."""
            if self.normalization != "zscore":
                v = v * (self.data_max - self.data_min) + self.data_min
                v = v / self.data_raw_std
            return v

        vnorm = mcolors.Normalize(vmin=self.vis_vmin, vmax=self.vis_vmax, clip=True)

        fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 8))
        if n == 1:
            axes = axes[:, None]

        for col, f in enumerate(frame_indices):
            gt_v = to_vis(gt_np[f, s, 0])
            gt_img = _VORTICITY_CMAP(vnorm(gt_v))[..., :3]
            axes[0, col].imshow(np.clip(gt_img, 0, 1))
            axes[0, col].set_title(f"t={f}", fontsize=9)
            axes[0, col].axis("off")

            if isinstance(self.obs_operator, SuperResolution):
                y_frame = y_np[f, s, 0]
                y_up = torch.nn.functional.interpolate(
                    torch.from_numpy(y_frame).float().unsqueeze(0).unsqueeze(0),
                    size=(gt_np.shape[-2], gt_np.shape[-1]),
                    mode="nearest",
                ).squeeze().numpy()
                obs_v = to_vis(y_up)
                obs_img = _VORTICITY_CMAP(vnorm(obs_v))[..., :3]
            elif isinstance(self.obs_operator, SparseObservation):
                obs_v = to_vis(y_np[f, s, 0].copy())
                mask_np = self.obs_operator.mask.cpu().numpy().squeeze()
                obs_v[mask_np < 0.5] = 0.0
                obs_img = _VORTICITY_CMAP(vnorm(obs_v))[..., :3]
            else:
                obs_v = to_vis(y_np[f, s, 0])
                obs_img = _VORTICITY_CMAP(vnorm(obs_v))[..., :3]

            axes[1, col].imshow(np.clip(obs_img, 0, 1))
            axes[1, col].axis("off")

            k_gt, spec_gt = self._energy_spectrum(gt_v)
            obs_v_clean = to_vis(gt_np[f, s, 0])
            k_obs, spec_obs = self._energy_spectrum(obs_v_clean)
            axes[2, col].loglog(k_gt, spec_gt, "b-", linewidth=1, label="GT")
            axes[2, col].loglog(k_obs, spec_obs, "r--", linewidth=1, label="GT (ref)")
            axes[2, col].set_xticks([])
            axes[2, col].set_yticks([])
            if col == 0:
                axes[2, col].legend(fontsize=7, loc="upper right")

        axes[0, 0].set_ylabel("GT", fontsize=12, rotation=0, labelpad=30, va="center")
        axes[1, 0].set_ylabel("Obs y", fontsize=12, rotation=0, labelpad=30, va="center")
        axes[2, 0].set_ylabel("E(k)", fontsize=12, rotation=0, labelpad=30, va="center")
        fig.suptitle(f"observation guidance Observation ({self.obs_operator})", fontsize=13, y=1.0)
        fig.tight_layout()

        logger.log({f"obs_obs/batch_{batch_idx}": wandb.Image(fig)})
        plt.close(fig)

    def _log_nrmse_curve(self, pred, gt, logger, batch_idx, sample_idx):
        """
        Log per-frame RMSE and NRMSE curves for one trajectory.

        RMSE(t)  = sqrt(mean((pred_t - gt_t)^2))   over (C, H, W)
        NRMSE(t) = ||pred_t - gt_t||_2 / ||gt_t||_2

        pred, gt: (n_frames, batch, C, H, W) in unnormalized data space.
        """
        p = pred[:, sample_idx].detach().cpu().float()  # (T, C, H, W)
        g = gt[:, sample_idx].detach().cpu().float()

        n_frames = p.shape[0]
        rmse  = np.zeros(n_frames)
        nrmse = np.zeros(n_frames)
        for t in range(n_frames):
            rmse[t] = torch.nn.functional.mse_loss(p[t], g[t]).item() ** 0.5
            gt_norm = torch.linalg.norm(g[t]).item()
            nrmse[t] = torch.linalg.norm(p[t] - g[t]).item() / gt_norm if gt_norm > 0 else 0.0

        print(f"[batch={batch_idx} sample={sample_idx}] RMSE per frame: {rmse}")
        print(f"[batch={batch_idx} sample={sample_idx}] NRMSE per frame: {nrmse}")

        def _make_fig(values, ylabel):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(range(n_frames), values, "o-", markersize=3, linewidth=1.5)
            ax.axvline(x=self.context_frames - 0.5, color="gray", linestyle="--",
                       linewidth=0.8, label="context boundary")
            ax.set_xlabel("Frame index")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs Frame — batch {batch_idx} sample {sample_idx}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            return fig

        fig_rmse = _make_fig(rmse, "RMSE")
        logger.log({f"rmse_curve/batch_{batch_idx}_sample_{sample_idx}": wandb.Image(fig_rmse)})
        plt.close(fig_rmse)

        fig_nrmse = _make_fig(nrmse, "NRMSE")
        logger.log({f"nrmse_curve/batch_{batch_idx}_sample_{sample_idx}": wandb.Image(fig_nrmse)})
        plt.close(fig_nrmse)

    def on_validation_epoch_end(self, namespace="validation"):
        if not self.validation_step_outputs:
            return

        xs_pred_list, xs_list = [], []
        for pred, gt in self.validation_step_outputs:
            xs_pred_list.append(pred)
            xs_list.append(gt)
        xs_pred = torch.cat(xs_pred_list, 1)
        xs = torch.cat(xs_list, 1)

        try:
            print(f"[NS-debug] val xs_pred  range [{xs_pred.min().item():.4f}, {xs_pred.max().item():.4f}]  mean={xs_pred.mean().item():.4f}")
            print(f"[NS-debug] val xs (GT)  range [{xs.min().item():.4f}, {xs.max().item():.4f}]  mean={xs.mean().item():.4f}")
            print(f"[NS-debug] normalization={self.normalization}, data_min={self.data_min}, data_max={self.data_max}, vis_vmin={self.vis_vmin}, vis_vmax={self.vis_vmax}, metrics_data_range={self._metrics_data_range:.4f}")

            xs_pred_rgb = self._vorticity_to_rgb(xs_pred)
            xs_rgb = self._vorticity_to_rgb(xs)

            log_video(
                xs_pred_rgb,
                xs_rgb,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )
        except Exception as e:
            print(f"[DiffusionForcingNS] Warning: validation video logging failed: {e}")

        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames :],
            xs[self.context_frames :],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
            data_range=self._metrics_data_range,
        )

        if namespace == "test":
            try:
                spec_dict = compute_spectrum_error_bands(
                    xs_pred[self.context_frames :],
                    xs[self.context_frames :],
                    mode="kinetic_energy",
                )
                metric_dict = {**metric_dict, **spec_dict}
            except Exception as e:
                print(f"[DiffusionForcingNS] Warning: spectrum error metric failed: {e}")

        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if namespace == "test":
            try:
                import hydra
                out_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
                save_dir = out_dir / "test_predictions"
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save(xs_pred.cpu(), save_dir / "xs_pred.pt")
                torch.save(xs.cpu(), save_dir / "xs_gt.pt")
                if self.obs_enabled and hasattr(self, "_obs_observations") and self._obs_observations:
                    y_all = torch.cat(self._obs_observations, dim=1)
                    torch.save(y_all.cpu(), save_dir / "y_obs.pt")
                print(f"[DiffusionForcingNS] Saved test predictions to {save_dir}/")
                print(f"  xs_pred.pt: {tuple(xs_pred.shape)}")
                print(f"  xs_gt.pt:   {tuple(xs.shape)}")
                print(f"  normalization: {self.normalization}")
                if self.normalization == "zscore":
                    print(f"  Values in raw/std space. To get original scale: v = x * {self.data_raw_std}")
                else:
                    print(f"  Values in [0,1]. To get original scale: v = x * {self.data_max - self.data_min:.2f} + ({self.data_min})")
            except Exception as e:
                print(f"[DiffusionForcingNS] Warning: failed to save test predictions: {e}")

        self.validation_step_outputs.clear()
        if hasattr(self, "_obs_observations"):
            self._obs_observations.clear()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(namespace="test")
