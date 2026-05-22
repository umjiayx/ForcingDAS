"""
Diffusion Forcing for ERA5 multi-variable weather data.

Extends DiffusionForcingVideo with:
  - Per-variable visualization (4-channel → 4-row colormap grids)
  - Weather-specific metrics: per-variable RMSE, ACC (anomaly correlation)
  - z-score normalization (data already normalized; data_mean=0, data_std=1)
  - observation guidance for data assimilation at test time
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
import wandb

from .df_video import DiffusionForcingVideo
from .df_base import DiffusionForcingBase
from .operators import get_operator
from utils.logging_utils import get_validation_metrics_for_videos

VAR_NAMES = ["z500", "t850", "u10", "v10"]
VAR_CMAPS = ["RdBu_r", "RdYlBu_r", "coolwarm", "coolwarm"]
VAR_UNITS = ["m²/s²", "K", "m/s", "m/s"]


class DiffusionForcingERA5(DiffusionForcingVideo):
    """
    DiffusionForcingVideo subclass for 4-channel ERA5 weather data.

    Data is already z-score normalized per channel (mean~0, std~1).
    _normalize_x is identity (data_mean=0, data_std=1).
    clip_noise is raised to accommodate the ~[-6, 6] data range.
    """

    def __init__(self, cfg: DictConfig):
        self.train_vis_every = cfg.train_vis_every

        # Data is already z-score normalized: _normalize_x should be identity
        cfg.data_mean = 0.0
        cfg.data_std = 1.0
        cfg.diffusion.clip_noise = max(cfg.diffusion.clip_noise, 15.0)

        # Per-channel raw statistics for un-normalizing to physical units
        stats_path = cfg.get("stats_path", "")
        if stats_path:
            stats = torch.load(stats_path, map_location="cpu")
            self._raw_mean = stats["mean"]  # (C,)
            self._raw_std = stats["std"]    # (C,)
            self._var_names = stats.get("var_names", VAR_NAMES)
        else:
            self._raw_mean = torch.zeros(4)
            self._raw_std = torch.ones(4)
            self._var_names = VAR_NAMES

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

        super().__init__(cfg)

    # ── Visualization ────────────────────────────────────────────────

    def _era5_to_rgb_grid(self, x, sample_idx=0):
        """
        Convert 4-channel ERA5 tensor to a list of per-variable RGB images.

        Args:
            x: (frame, batch, 4, H, W) in z-score normalized space.
            sample_idx: which batch element to visualize.
        Returns:
            list of 4 numpy arrays, each (n_frames, H, W, 3) in [0, 1].
        """
        v = x[:, sample_idx].detach().cpu().numpy()  # (F, 4, H, W)
        n_channels = v.shape[1]
        per_var_rgb = []
        for c in range(n_channels):
            ch = v[:, c]  # (F, H, W)
            vmin, vmax = ch.min(), ch.max()
            if abs(vmax - vmin) < 1e-8:
                vmin, vmax = -1.0, 1.0
            cmap = plt.colormaps.get_cmap(VAR_CMAPS[c] if c < len(VAR_CMAPS) else "viridis")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(ch))  # (F, H, W, 4)
            rgb = rgba[..., :3]    # (F, H, W, 3)
            rgb = rgb.transpose(0, 2, 1, 3)  # (F, W, H, 3) — landscape
            per_var_rgb.append(rgb)
        return per_var_rgb

    def _log_frame_grid(self, pred, gt, logger, namespace,
                        step=None, frame_stride=4, sample_idx=0,
                        title_prefix="", save_path=None):
        """Log a (3*C)-row (GT/Pred/Error per variable) x N-column grid to wandb and optionally save PNG."""
        n_frames = gt.shape[0]
        n_channels = gt.shape[2]
        frame_indices = list(range(0, n_frames, frame_stride))
        if (n_frames - 1) not in frame_indices:
            frame_indices.append(n_frames - 1)
        n = len(frame_indices)

        pred_rgb = self._era5_to_rgb_grid(pred, sample_idx)
        gt_rgb = self._era5_to_rgb_grid(gt, sample_idx)

        p_np = pred[:, sample_idx].detach().cpu().numpy()  # (F, C, H, W)
        g_np = gt[:, sample_idx].detach().cpu().numpy()
        err = np.abs(p_np - g_np)

        n_rows = n_channels * 3  # GT + Pred + Error for each variable
        fig, axes = plt.subplots(n_rows, n, figsize=(2.5 * n, 1.8 * n_rows))
        if n == 1:
            axes = axes[:, None]

        for c in range(n_channels):
            var_name = self._var_names[c] if c < len(self._var_names) else f"ch{c}"
            r_gt = c * 3
            r_pred = c * 3 + 1
            r_err = c * 3 + 2
            err_vmax = max(np.percentile(err[:, c], 98), 1e-6)

            for col, f in enumerate(frame_indices):
                axes[r_gt, col].imshow(np.clip(gt_rgb[c][f], 0, 1))
                axes[r_gt, col].axis("off")
                if col == 0:
                    axes[r_gt, col].set_ylabel(f"{var_name}\nGT", fontsize=8,
                                               rotation=0, labelpad=40, va="center")
                if c == 0:
                    axes[r_gt, col].set_title(f"t={f}", fontsize=9)

                axes[r_pred, col].imshow(np.clip(pred_rgb[c][f], 0, 1))
                axes[r_pred, col].axis("off")
                if col == 0:
                    axes[r_pred, col].set_ylabel(f"{var_name}\nPred", fontsize=8,
                                                 rotation=0, labelpad=40, va="center")

                im = axes[r_err, col].imshow(
                    err[f, c].T, cmap="hot", vmin=0, vmax=err_vmax,
                    aspect="auto", origin="lower",
                )
                axes[r_err, col].axis("off")
                if col == 0:
                    axes[r_err, col].set_ylabel(f"{var_name}\nError", fontsize=8,
                                                rotation=0, labelpad=40, va="center")

            fig.colorbar(im, ax=axes[r_err, :].tolist(), shrink=0.8, pad=0.02)

        sup_title = f"{title_prefix}ERA5 GT / Pred / |Error|" if title_prefix else "ERA5 GT / Pred / |Error|"
        fig.suptitle(sup_title, fontsize=13, y=1.0)
        fig.tight_layout()

        log_key = f"{namespace}/frame_grid"
        if step is not None:
            log_key = f"{log_key}_{step}"
        logger.log({log_key: wandb.Image(fig)})
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Weather metrics ──────────────────────────────────────────────

    def _compute_per_var_metrics(self, pred, gt, namespace):
        """
        Compute per-variable RMSE, NRMSE, and ACC in z-score normalized space.
        pred, gt: (frame, batch, C, H, W).
        """
        pred = pred.detach().cpu().float()
        gt = gt.detach().cpu().float()

        n_channels = gt.shape[2]
        metrics = {}
        for c in range(n_channels):
            var_name = self._var_names[c] if c < len(self._var_names) else f"ch{c}"
            p = pred[:, :, c]  # (F, B, H, W)
            g = gt[:, :, c]

            mse = (p - g).pow(2).mean()
            rmse = mse.sqrt().item()
            gt_norm = g.pow(2).mean().sqrt().item()
            nrmse = rmse / gt_norm if gt_norm > 0 else 0.0

            # ACC: anomaly correlation coefficient
            # In z-score space, "climatology" is 0 (the mean), so anomaly = value itself
            p_flat = p.reshape(-1)
            g_flat = g.reshape(-1)
            p_mean = p_flat.mean()
            g_mean = g_flat.mean()
            p_anom = p_flat - p_mean
            g_anom = g_flat - g_mean
            num = (p_anom * g_anom).sum()
            denom = (p_anom.pow(2).sum() * g_anom.pow(2).sum()).sqrt()
            acc = (num / denom).item() if denom > 0 else 0.0

            metrics[f"{var_name}_rmse"] = rmse
            metrics[f"{var_name}_nrmse"] = nrmse
            metrics[f"{var_name}_acc"] = acc

        # Aggregate metrics
        rmse_vals = [metrics[f"{self._var_names[c]}_rmse"] for c in range(n_channels)]
        acc_vals = [metrics[f"{self._var_names[c]}_acc"] for c in range(n_channels)]
        metrics["rmse_mean"] = float(np.mean(rmse_vals))
        metrics["acc_mean"] = float(np.mean(acc_vals))

        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return metrics

    def _log_nrmse_curve(self, pred, gt, logger, batch_idx, sample_idx):
        """Log per-frame, per-variable RMSE and NRMSE curves. Returns (n_frames, n_vars) NRMSE array."""
        p = pred[:, sample_idx].detach().cpu().float()  # (T, C, H, W)
        g = gt[:, sample_idx].detach().cpu().float()
        n_frames = p.shape[0]
        n_channels = p.shape[1]

        all_nrmse = np.zeros((n_frames, n_channels))

        fig_rmse, axes_rmse = plt.subplots(1, n_channels, figsize=(4 * n_channels, 3.5), sharey=True)
        fig_nrmse, axes_nrmse = plt.subplots(1, n_channels, figsize=(4 * n_channels, 3.5), sharey=True)
        if n_channels == 1:
            axes_rmse = [axes_rmse]
            axes_nrmse = [axes_nrmse]

        for c in range(n_channels):
            var_name = self._var_names[c] if c < len(self._var_names) else f"ch{c}"
            rmse = np.zeros(n_frames)
            nrmse = np.zeros(n_frames)
            for t in range(n_frames):
                rmse[t] = F.mse_loss(p[t, c], g[t, c]).item() ** 0.5
                gt_norm = torch.linalg.norm(g[t, c]).item()
                nrmse[t] = torch.linalg.norm(p[t, c] - g[t, c]).item() / gt_norm if gt_norm > 0 else 0.0

            all_nrmse[:, c] = nrmse

            axes_rmse[c].plot(range(n_frames), rmse, "o-", markersize=3, linewidth=1.5)
            axes_rmse[c].axvline(x=self.context_frames - 0.5, color="gray", linestyle="--",
                                  linewidth=0.8, label="context")
            axes_rmse[c].set_title(var_name, fontsize=10)
            axes_rmse[c].set_xlabel("Frame")
            axes_rmse[c].grid(True, alpha=0.3)
            if c == 0:
                axes_rmse[c].set_ylabel("RMSE")
                axes_rmse[c].legend(fontsize=7)

            axes_nrmse[c].plot(range(n_frames), nrmse, "o-", markersize=3, linewidth=1.5)
            axes_nrmse[c].axvline(x=self.context_frames - 0.5, color="gray", linestyle="--",
                                   linewidth=0.8, label="context")
            axes_nrmse[c].set_title(var_name, fontsize=10)
            axes_nrmse[c].set_xlabel("Frame")
            axes_nrmse[c].grid(True, alpha=0.3)
            if c == 0:
                axes_nrmse[c].set_ylabel("NRMSE")
                axes_nrmse[c].legend(fontsize=7)

        fig_rmse.suptitle(f"Per-variable RMSE — batch {batch_idx} sample {sample_idx}", fontsize=11)
        fig_rmse.tight_layout()
        logger.log({f"rmse_curve/batch_{batch_idx}_sample_{sample_idx}": wandb.Image(fig_rmse)})
        plt.close(fig_rmse)

        fig_nrmse.suptitle(f"Per-variable NRMSE — batch {batch_idx} sample {sample_idx}", fontsize=11)
        fig_nrmse.tight_layout()
        logger.log({f"nrmse_curve/batch_{batch_idx}_sample_{sample_idx}": wandb.Image(fig_nrmse)})
        plt.close(fig_nrmse)

        return torch.from_numpy(all_nrmse)  # (n_frames, n_channels)

    # ── observation guidance measurement functions ────────────────────────────────────

    def _make_measurement_fn(self, y_obs, from_noise_levels, to_noise_levels):
        """Build a differentiable measurement loss for one DDIM step."""
        active = (from_noise_levels > to_noise_levels)
        sigma_y = self.obs_noise_sigma
        operator = self.obs_operator

        def measurement_fn(x_hat_0):
            x_data = self._unnormalize_x(x_hat_0)
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
        """Measurement loss with SDA variance reweighting (gamma > 0)."""
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

    # ── observation guidance validation step ──────────────────────────────────────────

    @torch.no_grad()
    def _obs_validation_step(self, batch, batch_idx, namespace="test"):
        """Sampling loop with observation guidance guidance."""
        xs, conditions, masks = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape

        xs_unnorm = self._unnormalize_x(xs)
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
                if getattr(self, 'obs_gamma', 0) > 0:
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

    # ── Training / validation / test overrides ───────────────────────

    def training_step(self, batch, batch_idx):
        output_dict = DiffusionForcingBase.training_step(self, batch, batch_idx)

        if self.global_step > 0 and self.global_step % self.train_vis_every == 0:
            try:
                self._log_frame_grid(
                    output_dict["xs_pred"], output_dict["xs"],
                    logger=self.logger.experiment,
                    namespace="training_vis",
                    step=self.global_step,
                    frame_stride=max(1, output_dict["xs"].shape[0] // 6),
                )
            except Exception as e:
                if not getattr(self, "_video_log_warned", False):
                    print(f"[DiffusionForcingERA5] Warning: training vis failed: {e}")
                    self._video_log_warned = True

        return output_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        if self.obs_enabled:
            loss = self._obs_validation_step(batch, batch_idx, namespace="test")
        else:
            loss = self.validation_step(batch, batch_idx, *args, **kwargs, namespace="test")

        pred, gt = self.validation_step_outputs[-1]
        try:
            import hydra as _hydra
            out_dir = Path(_hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
        except Exception:
            out_dir = None

        try:
            bi = len(self.validation_step_outputs) - 1
            n_samples = pred.shape[1]
            for s in range(n_samples):
                png_path = None
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    png_path = str(out_dir / f"frames_batch_{bi}_sample_{s}.png")
                self._log_frame_grid(
                    pred, gt,
                    logger=self.logger.experiment,
                    namespace=f"test_frames/batch_{bi}",
                    step=s,
                    frame_stride=max(1, pred.shape[0] // 8),
                    sample_idx=s,
                    save_path=png_path,
                )

            if self.obs_enabled:
                if not hasattr(self, "_nrmse_per_frame_list"):
                    self._nrmse_per_frame_list = []
                for s in range(n_samples):
                    nrmse_tensor = self._log_nrmse_curve(
                        pred, gt,
                        logger=self.logger.experiment,
                        batch_idx=bi,
                        sample_idx=s,
                    )
                    self._nrmse_per_frame_list.append(nrmse_tensor)
            print(f"[DiffusionForcingERA5] Logged test batch {bi} to wandb")
        except Exception as e:
            print(f"[DiffusionForcingERA5] Warning: per-batch test vis failed: {e}")

        return loss

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
            self._log_frame_grid(
                xs_pred, xs,
                logger=self.logger.experiment,
                namespace=namespace + "_vis",
                step=None if namespace == "test" else self.global_step,
                frame_stride=max(1, xs.shape[0] // 8),
            )
        except Exception as e:
            print(f"[DiffusionForcingERA5] Warning: validation vis failed: {e}")

        # Per-variable metrics (on predicted frames, excluding context)
        self._compute_per_var_metrics(
            xs_pred[self.context_frames:],
            xs[self.context_frames:],
            namespace=namespace,
        )

        # Standard video metrics (MSE, PSNR, SSIM)
        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames:],
            xs[self.context_frames:],
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(self.validation_fvd_model[0] if self.validation_fvd_model else None),
            data_range=12.0,  # z-score data range ~[-6, 6]
        )
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
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(xs_pred.cpu(), out_dir / "xs_pred.pt")
                torch.save(xs.cpu(), out_dir / "xs_gt.pt")
                if self.obs_enabled and hasattr(self, "_obs_observations") and self._obs_observations:
                    y_all = torch.cat(self._obs_observations, dim=1)
                    torch.save(y_all.cpu(), out_dir / "y_obs.pt")
                if hasattr(self, "_nrmse_per_frame_list") and self._nrmse_per_frame_list:
                    nrmse_stack = torch.stack(self._nrmse_per_frame_list, dim=0)  # (n_batches, n_frames, n_vars)
                    torch.save({
                        "nrmse": nrmse_stack,
                        "var_names": self._var_names,
                    }, out_dir / "nrmse_per_frame.pt")
                print(f"[DiffusionForcingERA5] Saved test outputs to {out_dir}/")
            except Exception as e:
                print(f"[DiffusionForcingERA5] Warning: failed to save test predictions: {e}")

        self.validation_step_outputs.clear()
        if hasattr(self, "_obs_observations"):
            self._obs_observations.clear()
        if hasattr(self, "_nrmse_per_frame_list"):
            self._nrmse_per_frame_list.clear()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(namespace="test")
