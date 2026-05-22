"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion


class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim
        self.causal = cfg.causal

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise

        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay ** (self.frame_stack * cfg.frame_skip)

        self.validation_step_outputs = []
        super().__init__(cfg)

    def _build_model(self):
        self.diffusion_model = Diffusion(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        noise_levels, zero_noise_mask = self._generate_noise_levels(xs)
        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=noise_levels, zero_noise_mask=zero_noise_mask)
        loss = self.reweight_loss(loss, masks)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            curr_frame += horizon
            pbar.update(horizon)

        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """
        Generate per-frame noise levels for training.

        noise_levels is a (T, B) matrix: each column is one sample, each row is
        one frame.  The modes build progressively tighter matches to the noise
        pattern the model sees at inference.

        Example (T=6, B=1, timesteps=1000, sampling_timesteps=50, stabilization_level=15):

          random_all — i.i.d. uniform, no structure:
              [731, 102, 955,  47, 503, 288]

          random_causal — sort along the frame axis (ascending):
              [ 47, 102, 288, 503, 731, 955]
              Earlier frames are cleaner, later frames noisier — matching the
              monotonic staircase seen during autoregressive/pyramid sampling.

          random_causal + context_clean_ratio (n_ctx=2):
              [ 14,  14, 288, 503, 731, 955]
               ↑────↑
               context frames set to stabilization_level-1 with zero noise,
               exactly replicating the sampling code path.

          random_causal_exact_pyramid — prediction frames follow the exact
              pyramid staircase (uncertainty_scale=1), mapped from DDIM space
              to real timesteps (step size = timesteps / sampling_timesteps = 20):
              [ 14,  14,  99, 119, 139, 159]
               ↑────↑     ↑── uniform step of 20, not random gaps

          random_causal_exact_autoregressive — same staircase formula but with
              uncertainty_scale=ST.  At most one prediction frame is "active"
              (intermediate level); the rest are either done (0) or max noise:
              [731, 102,   0, 499, 999, 999]
               ↑────↑     done  ↑    ↑── max noise (not yet started)
               random     active frame

          random_causal_exact_both — per sample, coin-flip (pyramid_ratio)
              between exact pyramid and exact autoregressive:
              sample A (pyramid):         [ 14,  14,  99, 119, 139, 159]
              sample B (autoregressive):  [731, 102,   0, 499, 999, 999]

        Returns:
            noise_levels: (T, B) integer tensor in [0, timesteps)
            zero_noise_mask: (T, B) bool tensor or None — True means use zero
                noise (replicating the sampling behavior for context frames)
        """
        num_frames, batch_size, *_ = xs.shape
        zero_noise_mask = None

        match self.cfg.noise_level:
            # ── Baseline: every frame gets an independent uniform noise level ──
            case "random_all":
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

            # ── CAT: with prob causal_ratio, sort noise levels into a monotone
            #    staircase (clean past → noisy future).  Optionally clean context
            #    frames to match inference exactly. ──
            case "random_causal":
                # Step 1: start with i.i.d. uniform noise levels, same as random_all
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

                # Step 2: with prob causal_ratio, select a subset of samples for monotone sorting
                causal_mask = torch.rand(batch_size, device=xs.device) < self.cfg.causal_ratio
                if causal_mask.any():
                    # Sort noise levels along time axis → non-decreasing (clean past, noisy future)
                    sorted_levels, _ = torch.sort(noise_levels[:, causal_mask], dim=0)
                    noise_levels[:, causal_mask] = sorted_levels

                    # Step 3: optionally replicate exact sampling behavior for context frames
                    if self.cfg.context_clean_ratio > 0:
                        # Upper bound for random context length (dedicated config or dataset default)
                        if self.cfg.max_context_frames > 0:
                            max_n_ctx = self.cfg.max_context_frames // self.frame_stack
                        else:
                            max_n_ctx = self.context_frames // self.frame_stack

                        # Within causal samples, further select a subset for clean context
                        ctx_mask = torch.rand(causal_mask.sum(), device=xs.device) < self.cfg.context_clean_ratio
                        if ctx_mask.any() and max_n_ctx > 0:
                            causal_indices = causal_mask.nonzero(as_tuple=True)[0]
                            ctx_batch_indices = causal_indices[ctx_mask]

                            # Random context length so the model generalizes to any context_length at inference
                            n_ctx = torch.randint(1, max_n_ctx + 1, (1,), device=xs.device).item()

                            # Set context to stabilization_level-1 + zero noise (exact sampling match)
                            stab_level = self.cfg.diffusion.stabilization_level - 1
                            noise_levels[:n_ctx, ctx_batch_indices] = stab_level
                            zero_noise_mask = torch.zeros(num_frames, batch_size, dtype=torch.bool, device=xs.device)
                            zero_noise_mask[:n_ctx, ctx_batch_indices] = True

            # ── Exact pyramid: with prob causal_ratio, replicate the pyramid
            #    sampling schedule (uncertainty_scale=1) during training.  A random
            #    denoising step is sampled per batch element, producing the exact
            #    staircase the model will see at inference.  Context frames are
            #    optionally cleaned with prob context_clean_ratio. ──
            case "random_causal_exact_pyramid":
                # Step 1: start with i.i.d. uniform (for non-causal samples)
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

                # Step 2: select samples for pyramid staircase
                causal_mask = torch.rand(batch_size, device=xs.device) < self.cfg.causal_ratio
                if causal_mask.any():
                    n_causal = causal_mask.sum().item()
                    causal_indices = causal_mask.nonzero(as_tuple=True)[0]
                    ST = self.sampling_timesteps  # DDIM steps (e.g. 50)

                    # Random context length
                    if self.cfg.max_context_frames > 0:
                        max_n_ctx = self.cfg.max_context_frames // self.frame_stack
                    else:
                        max_n_ctx = self.context_frames // self.frame_stack
                    n_ctx = torch.randint(1, max_n_ctx + 1, (1,), device=xs.device).item()
                    n_pred = num_frames - n_ctx

                    # Step 3: build pyramid staircase for prediction frames
                    # Pyramid formula: ddim_level[t] = clip(ST + t * us - m, 0, ST)
                    # with uncertainty_scale=1 (fixed for this mode)
                    height = ST + n_pred  # = ST + (n_pred - 1) * 1 + 1
                    m = torch.randint(0, height, (n_causal,), device=xs.device)

                    frame_idx = torch.arange(n_pred, device=xs.device)
                    ddim_levels = (ST + frame_idx.unsqueeze(1) - m.unsqueeze(0)).clamp(0, ST)

                    # Map DDIM step indices → real timesteps (same mapping used in sampling)
                    real_steps = torch.linspace(-1, self.timesteps - 1, steps=ST + 1, device=xs.device).long()
                    pred_noise = real_steps[ddim_levels].clamp(0, self.timesteps - 1)
                    # Note: when n_pred > ST, trailing frames may saturate at the max
                    # noise level for early denoising steps — this is consistent with
                    # actual pyramid sampling where those frames haven't begun denoising.

                    noise_levels[n_ctx:, causal_indices] = pred_noise

                    # NOTE (small train-test gap): prediction frames whose ddim_level
                    # lands at 0 are mapped to real timestep 0 and receive *real* Gaussian
                    # noise here, whereas during inference they would be treated like
                    # context (stabilization_level-1 + zero noise).  This gap is negligible
                    # when the full sequence is rolled out at once (n_frames = window),
                    # because there is no iterative chunk handoff where the mismatch could
                    # compound.  It only matters for chunked pyramid generation with
                    # sliding windows — if that becomes a use-case, extend zero_noise_mask
                    # to cover these "finished" prediction frames as well.

                    # Step 4: optionally clean context frames (same mechanism as random_causal)
                    if self.cfg.context_clean_ratio > 0 and n_ctx > 0:
                        ctx_mask = torch.rand(n_causal, device=xs.device) < self.cfg.context_clean_ratio
                        if ctx_mask.any():
                            ctx_batch_indices = causal_indices[ctx_mask]
                            stab_level = self.cfg.diffusion.stabilization_level - 1
                            noise_levels[:n_ctx, ctx_batch_indices] = stab_level
                            zero_noise_mask = torch.zeros(num_frames, batch_size, dtype=torch.bool, device=xs.device)
                            zero_noise_mask[:n_ctx, ctx_batch_indices] = True

            # ── Exact autoregressive: same formula as exact pyramid but with
            #    uncertainty_scale=ST, so at most one prediction frame is "active"
            #    per denoising step — matching autoregressive sampling. ──
            case "random_causal_exact_autoregressive":
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

                causal_mask = torch.rand(batch_size, device=xs.device) < self.cfg.causal_ratio
                if causal_mask.any():
                    n_causal = causal_mask.sum().item()
                    causal_indices = causal_mask.nonzero(as_tuple=True)[0]
                    ST = self.sampling_timesteps

                    if self.cfg.max_context_frames > 0:
                        max_n_ctx = self.cfg.max_context_frames // self.frame_stack
                    else:
                        max_n_ctx = self.context_frames // self.frame_stack
                    n_ctx = torch.randint(1, max_n_ctx + 1, (1,), device=xs.device).item()
                    n_pred = num_frames - n_ctx

                    # Autoregressive: ddim_level[t] = clip(ST + t * ST - m, 0, ST)
                    height = ST * n_pred + 1
                    m = torch.randint(0, height, (n_causal,), device=xs.device)

                    frame_idx = torch.arange(n_pred, device=xs.device)
                    ddim_levels = (ST + frame_idx.unsqueeze(1) * ST - m.unsqueeze(0)).clamp(0, ST)

                    real_steps = torch.linspace(-1, self.timesteps - 1, steps=ST + 1, device=xs.device).long()
                    pred_noise = real_steps[ddim_levels].clamp(0, self.timesteps - 1)

                    noise_levels[n_ctx:, causal_indices] = pred_noise

                    # NOTE (small train-test gap): same as random_causal_exact_pyramid —
                    # prediction frames at ddim_level=0 get real noise instead of zero
                    # noise.  Negligible for full-sequence rollout.

                    if self.cfg.context_clean_ratio > 0 and n_ctx > 0:
                        ctx_mask = torch.rand(n_causal, device=xs.device) < self.cfg.context_clean_ratio
                        if ctx_mask.any():
                            ctx_batch_indices = causal_indices[ctx_mask]
                            stab_level = self.cfg.diffusion.stabilization_level - 1
                            noise_levels[:n_ctx, ctx_batch_indices] = stab_level
                            zero_noise_mask = torch.zeros(num_frames, batch_size, dtype=torch.bool, device=xs.device)
                            zero_noise_mask[:n_ctx, ctx_batch_indices] = True

            # ── Exact both: with prob causal_ratio, apply an inference-like
            #    schedule; within that, coin-flip (pyramid_ratio) between exact
            #    pyramid (us=1) and exact autoregressive (us=ST).  The remaining
            #    (1 - causal_ratio) fraction sees random_all. ──
            case "random_causal_exact_both":
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

                causal_mask = torch.rand(batch_size, device=xs.device) < self.cfg.causal_ratio
                if causal_mask.any():
                    n_causal = causal_mask.sum().item()
                    causal_indices = causal_mask.nonzero(as_tuple=True)[0]
                    ST = self.sampling_timesteps

                    if self.cfg.max_context_frames > 0:
                        max_n_ctx = self.cfg.max_context_frames // self.frame_stack
                    else:
                        max_n_ctx = self.context_frames // self.frame_stack
                    n_ctx = torch.randint(1, max_n_ctx + 1, (1,), device=xs.device).item()
                    n_pred = num_frames - n_ctx

                    frame_idx = torch.arange(n_pred, device=xs.device)
                    real_steps = torch.linspace(-1, self.timesteps - 1, steps=ST + 1, device=xs.device).long()

                    # Split causal samples into pyramid and autoregressive groups
                    pyr_mask = torch.rand(n_causal, device=xs.device) < self.cfg.pyramid_ratio

                    # Pyramid group (us=1)
                    if pyr_mask.any():
                        n_pyr = pyr_mask.sum().item()
                        pyr_indices = causal_indices[pyr_mask]
                        height_pyr = ST + n_pred
                        m_pyr = torch.randint(0, height_pyr, (n_pyr,), device=xs.device)
                        ddim_pyr = (ST + frame_idx.unsqueeze(1) - m_pyr.unsqueeze(0)).clamp(0, ST)
                        pred_pyr = real_steps[ddim_pyr].clamp(0, self.timesteps - 1)
                        noise_levels[n_ctx:, pyr_indices] = pred_pyr

                    # Autoregressive group (us=ST)
                    ar_mask = ~pyr_mask
                    if ar_mask.any():
                        n_ar = ar_mask.sum().item()
                        ar_indices = causal_indices[ar_mask]
                        height_ar = ST * n_pred + 1
                        m_ar = torch.randint(0, height_ar, (n_ar,), device=xs.device)
                        ddim_ar = (ST + frame_idx.unsqueeze(1) * ST - m_ar.unsqueeze(0)).clamp(0, ST)
                        pred_ar = real_steps[ddim_ar].clamp(0, self.timesteps - 1)
                        noise_levels[n_ctx:, ar_indices] = pred_ar

                    # NOTE (small train-test gap): same as the other exact modes.

                    if self.cfg.context_clean_ratio > 0 and n_ctx > 0:
                        ctx_mask = torch.rand(n_causal, device=xs.device) < self.cfg.context_clean_ratio
                        if ctx_mask.any():
                            ctx_batch_indices = causal_indices[ctx_mask]
                            stab_level = self.cfg.diffusion.stabilization_level - 1
                            noise_levels[:n_ctx, ctx_batch_indices] = stab_level
                            zero_noise_mask = torch.zeros(num_frames, batch_size, dtype=torch.bool, device=xs.device)
                            zero_noise_mask[:n_ctx, ctx_batch_indices] = True

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels, zero_noise_mask

    def _generate_scheduling_matrix(self, horizon: int):
        match self.cfg.scheduling_matrix:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
