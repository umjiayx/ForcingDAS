"""
Diffusion Forcing for SEVIR-LR VIL precipitation data.

Extends DiffusionForcingVideo with:
  - SEVIR VIL colormap visualization (1-ch VIL → 3-ch RGB)
  - CSI (Critical Success Index) evaluation at standard thresholds
  - Configurable normalization scale (norm_scale):
      norm_scale=1 → data mapped to [-1, 1]  (default)
      norm_scale=6 → data mapped to [-6, 6]  (wider dynamic range)
  - observation guidance for inverse problems at test time
  - Saves raw predictions at test time
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def _vil_cmap():
    """SEVIR VIL colormap for encoded uint8 [0, 255] values."""
    cols = [
        [0, 0, 0],
        [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
        [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
        [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
        [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
        [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
        [0.9607843137254902, 0.9607843137254902, 0.0],
        [0.9294117647058824, 0.6745098039215687, 0.0],
        [0.9411764705882353, 0.43137254901960786, 0.0],
        [0.6274509803921569, 0.0, 0.0],
        [0.9058823529411765, 0.0, 1.0],
    ]
    lev = [16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap = mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap, norm


_VIL_CMAP, _VIL_NORM = _vil_cmap()


def compute_csi(pred_255: np.ndarray, gt_255: np.ndarray, thresholds):
    """
    Compute CSI at each threshold on arrays in [0, 255].

    Args:
        pred_255: (N, H, W) predictions in [0, 255].
        gt_255:   (N, H, W) ground truth in [0, 255].
        thresholds: list of VIL thresholds.

    Returns:
        dict  {f"CSI-{t}": float, ..., "CSI-mean": float}
    """
    results = {}
    csi_values = []
    for t in thresholds:
        p_bin = (pred_255 >= t).astype(np.float32)
        g_bin = (gt_255 >= t).astype(np.float32)
        hits = (p_bin * g_bin).sum()
        misses = ((1 - p_bin) * g_bin).sum()
        false_alarms = (p_bin * (1 - g_bin)).sum()
        denom = hits + misses + false_alarms
        csi = (hits / denom).item() if denom > 0 else 0.0
        results[f"CSI-{int(t)}"] = csi
        csi_values.append(csi)
    results["CSI-mean"] = float(np.mean(csi_values))
    return results


class DiffusionForcingSEVIR(DiffusionForcingVideo):
    """
    DiffusionForcingVideo subclass for 1-channel SEVIR VIL data.

    - VIL colormap visualization for wandb logging
    - CSI metric computation at standard thresholds
    """

    def __init__(self, cfg: DictConfig):
        self.csi_thresholds = list(cfg.csi_thresholds)
        self.train_vis_every = cfg.train_vis_every

        norm_scale = float(cfg.get("norm_scale", 1))
        self.norm_scale = norm_scale
        cfg.data_mean = 0.5
        cfg.data_std = 0.5 / norm_scale
        if norm_scale > 1:
            cfg.diffusion.clip_noise = max(cfg.diffusion.clip_noise, 2.5 * norm_scale)

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

    def _vil_to_rgb(self, x):
        """
        Convert 1-channel VIL tensor to 3-channel RGB using the SEVIR colormap.

        Args:
            x: (frame, batch, 1, H, W) after _unnormalize_x (values ≈ [0, 1]).
        Returns:
            (frame, batch, 3, H, W) in [0, 1].
        """
        v = x.detach().cpu().numpy()
        v = (v * 255.0).clip(0, 255)
        v = v.squeeze(2)  # (F, B, H, W)

        rgba = _VIL_CMAP(_VIL_NORM(v))  # (F, B, H, W, 4)
        rgb = rgba[..., :3]
        rgb = np.transpose(rgb, (0, 1, 4, 2, 3))  # (F, B, 3, H, W)
        return torch.from_numpy(rgb.astype(np.float32))

    def _compute_csi_metrics(self, xs_pred, xs, namespace):
        """Compute and log CSI metrics.  Inputs are after _unnormalize_x (≈ [0,1])."""
        pred_np = (xs_pred.detach().cpu().numpy() * 255.0).clip(0, 255)
        gt_np = (xs.detach().cpu().numpy() * 255.0).clip(0, 255)

        # (F, B, 1, H, W) -> (F*B, H, W)
        pred_flat = pred_np.squeeze(2).reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
        gt_flat = gt_np.squeeze(2).reshape(-1, gt_np.shape[-2], gt_np.shape[-1])

        csi_dict = compute_csi(pred_flat, gt_flat, self.csi_thresholds)
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in csi_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return csi_dict

    def _log_frame_grid(self, pred_rgb, gt_rgb, logger, namespace,
                        step=None, frame_stride=4, sample_idx=0):
        """Log a 2-row (GT / Pred) x N-column grid to wandb."""
        n_frames = gt_rgb.shape[0]
        frame_indices = list(range(0, n_frames, frame_stride))
        if (n_frames - 1) not in frame_indices:
            frame_indices.append(n_frames - 1)
        n = len(frame_indices)

        fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5))
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

        axes[0, 0].set_ylabel("GT", fontsize=12, rotation=0, labelpad=30, va="center")
        axes[1, 0].set_ylabel("Pred", fontsize=12, rotation=0, labelpad=30, va="center")

        fig.suptitle("VIL Frame Comparison", fontsize=13, y=1.0)
        fig.tight_layout()

        log_key = f"{namespace}/frame_grid"
        if step is not None:
            log_key = f"{log_key}_{step}"
        logger.log({log_key: wandb.Image(fig)})
        plt.close(fig)

    def _make_measurement_fn(self, y_obs, from_noise_levels, to_noise_levels):
        """
        Build a differentiable measurement loss for one DDIM step.

        Args:
            y_obs: observations for the window, shape (frames, batch, C, H', W')
                   in data space (after operator, before noise was added separately).
            from_noise_levels: (frames, batch) current noise levels.
            to_noise_levels: (frames, batch) target noise levels.

        Returns:
            callable: x_hat_0 (normalized) -> scalar loss
        """
        active = (from_noise_levels > to_noise_levels)  # (frames, batch)
        sigma_y = self.obs_noise_sigma
        operator = self.obs_operator

        def measurement_fn(x_hat_0):
            x_data = self._unnormalize_x(x_hat_0).clamp(0, 1)
            shape = x_data.shape  # (frames, batch, C, H, W)
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
            x_data = self._unnormalize_x(x_hat_0).clamp(0, 1)
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
            x_data = self._unnormalize_x(x_hat_0).clamp(0, 1)
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

        xs_unnorm = self._unnormalize_x(xs).clamp(0, 1)
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

        # Also store observations for logging
        if not hasattr(self, "_obs_observations"):
            self._obs_observations = []
        self._obs_observations.append(y_obs.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        output_dict = DiffusionForcingBase.training_step(self, batch, batch_idx)

        if self.global_step > 0 and self.global_step % self.train_vis_every == 0:
            try:
                xs_pred_rgb = self._vil_to_rgb(output_dict["xs_pred"])
                xs_rgb = self._vil_to_rgb(output_dict["xs"])
                log_video(
                    xs_pred_rgb,
                    xs_rgb,
                    step=self.global_step,
                    namespace="training_vis",
                    logger=self.logger.experiment,
                )
            except Exception as e:
                if not getattr(self, "_video_log_warned", False):
                    print(f"[DiffusionForcingSEVIR] Warning: training video logging failed: {e}")
                    self._video_log_warned = True

        return output_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        if self.obs_enabled:
            loss = self._obs_validation_step(batch, batch_idx, namespace="test")
        else:
            loss = self.validation_step(batch, batch_idx, *args, **kwargs, namespace="test")

        pred, gt = self.validation_step_outputs[-1]
        try:
            pred_rgb = self._vil_to_rgb(pred)
            gt_rgb = self._vil_to_rgb(gt)
            bi = len(self.validation_step_outputs) - 1
            n_samples = pred.shape[1]
            for s in range(n_samples):
                self._log_frame_grid(
                    pred_rgb, gt_rgb,
                    logger=self.logger.experiment,
                    namespace=f"test_frames/batch_{bi}",
                    step=s,
                    frame_stride=1,
                    sample_idx=s,
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
        except Exception as e:
            print(f"[DiffusionForcingSEVIR] Warning: per-batch test vis failed: {e}")

        return loss

    def _log_obs_observation(self, y_obs, gt, logger, batch_idx):
        """Log a visualization of the observation y alongside GT for the first sample."""
        from .operators import SuperResolution, SparseObservation

        y_np = y_obs.cpu().numpy()
        gt_np = gt.cpu().numpy()

        # y is in data space [0,1] (approximately), shape: (frames, batch, C, H_y, W_y)
        n_frames = gt_np.shape[0]
        frame_indices = list(range(0, n_frames, max(1, n_frames // 8)))
        if (n_frames - 1) not in frame_indices:
            frame_indices.append(n_frames - 1)
        n = len(frame_indices)
        s = 0  # first sample

        fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5))
        if n == 1:
            axes = axes[:, None]

        gt_255 = (gt_np[:, s, 0] * 255.0).clip(0, 255)

        for col, f in enumerate(frame_indices):
            gt_img = _VIL_CMAP(_VIL_NORM(gt_255[f]))[..., :3]
            axes[0, col].imshow(np.clip(gt_img, 0, 1))
            axes[0, col].set_title(f"t={f}", fontsize=9)
            axes[0, col].axis("off")

            if isinstance(self.obs_operator, SuperResolution):
                y_frame = y_np[f, s, 0]
                y_up = F.interpolate(
                    torch.from_numpy(y_frame).float().unsqueeze(0).unsqueeze(0),
                    size=(gt_np.shape[-2], gt_np.shape[-1]),
                    mode="nearest",
                ).squeeze().numpy()
                y_255 = (y_up * 255.0).clip(0, 255)
                y_img = _VIL_CMAP(_VIL_NORM(y_255))[..., :3]
            elif isinstance(self.obs_operator, SparseObservation):
                y_frame_255 = (y_np[f, s, 0] * 255.0).clip(0, 255)
                mask_np = self.obs_operator.mask.cpu().numpy().squeeze()
                y_frame_255[mask_np < 0.5] = 0.0
                y_img = _VIL_CMAP(_VIL_NORM(y_frame_255))[..., :3]
            else:
                y_frame_255 = (y_np[f, s, 0] * 255.0).clip(0, 255)
                y_img = _VIL_CMAP(_VIL_NORM(y_frame_255))[..., :3]

            axes[1, col].imshow(np.clip(y_img, 0, 1))
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("GT", fontsize=12, rotation=0, labelpad=30, va="center")
        axes[1, 0].set_ylabel("Obs y", fontsize=12, rotation=0, labelpad=30, va="center")
        fig.suptitle(f"observation guidance Observation ({self.obs_operator})", fontsize=13, y=1.0)
        fig.tight_layout()

        logger.log({f"obs_obs/batch_{batch_idx}": wandb.Image(fig)})
        plt.close(fig)

    def _log_nrmse_curve(self, pred, gt, logger, batch_idx, sample_idx):
        """
        Log per-frame RMSE and NRMSE curves for one trajectory.

        RMSE(t)  = sqrt(mean((pred_t - gt_t)^2))   over (C, H, W)
        NRMSE(t) = ||pred_t - gt_t||_2 / ||gt_t||_2

        Data is in unnormalized [0, 1] space.
        pred, gt: (n_frames, batch, C, H, W).
        """
        p = pred[:, sample_idx].detach().cpu().float()  # (T, C, H, W)
        g = gt[:, sample_idx].detach().cpu().float()

        n_frames = p.shape[0]
        rmse  = np.zeros(n_frames)
        nrmse = np.zeros(n_frames)
        for t in range(n_frames):
            rmse[t] = F.mse_loss(p[t], g[t]).item() ** 0.5
            gt_norm = torch.linalg.norm(g[t]).item()
            nrmse[t] = torch.linalg.norm(p[t] - g[t]).item() / gt_norm if gt_norm > 0 else 0.0

        print(f"[batch={batch_idx} sample={sample_idx}] RMSE per frame: {rmse}")
        print(f"[batch={batch_idx} sample={sample_idx}] NRMSE per frame: {nrmse}")

        def _make_fig(values, ylabel, title_suffix):
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

        fig_rmse = _make_fig(rmse, "RMSE", "RMSE")
        logger.log({f"rmse_curve/batch_{batch_idx}_sample_{sample_idx}": wandb.Image(fig_rmse)})
        plt.close(fig_rmse)

        fig_nrmse = _make_fig(nrmse, "NRMSE", "NRMSE")
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
            xs_pred_rgb = self._vil_to_rgb(xs_pred)
            xs_rgb = self._vil_to_rgb(xs)

            log_video(
                xs_pred_rgb,
                xs_rgb,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )
        except Exception as e:
            print(f"[DiffusionForcingSEVIR] Warning: validation video logging failed: {e}")

        # CSI metrics (only on predicted frames, excluding context)
        self._compute_csi_metrics(
            xs_pred[self.context_frames:],
            xs[self.context_frames:],
            namespace=namespace,
        )

        # Standard video metrics (PSNR, SSIM, etc.)
        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames:],
            xs[self.context_frames:],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
            data_range=1.0,
        )

        if namespace == "test":
            try:
                spec_dict = compute_spectrum_error_bands(
                    xs_pred[self.context_frames:],
                    xs[self.context_frames:],
                    mode="raw_power",
                )
                metric_dict = {**metric_dict, **spec_dict}
            except Exception as e:
                print(f"[DiffusionForcingSEVIR] Warning: spectrum error metric failed: {e}")

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
                print(f"[DiffusionForcingSEVIR] Saved test predictions to {save_dir}/")
            except Exception as e:
                print(f"[DiffusionForcingSEVIR] Warning: failed to save test predictions: {e}")

        self.validation_step_outputs.clear()
        if hasattr(self, "_obs_observations"):
            self._obs_observations.clear()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(namespace="test")
