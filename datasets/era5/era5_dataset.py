import torch
import numpy as np
from omegaconf import DictConfig
from pathlib import Path


class ERA5Dataset(torch.utils.data.Dataset):
    """
    ERA5 multi-variable weather dataset for diffusion forcing.

    Expects ``cfg.data_dir`` to point to a folder containing ``train/``,
    ``val/``, and ``test/`` subfolders, each holding a ``data.pt`` tensor
    of shape ``(B, T, C, H, W)`` with float32 values, **already z-score
    normalized per channel** using training statistics.

    A companion ``stats.pt`` file in the root provides the original
    per-channel mean and std (for un-normalizing back to physical units)
    as well as variable names and metadata.

    Channels (C=4): z500, t850, u10, v10.
    Grid (after preprocessing): 240 x 120 (lon x lat), ~1.5 deg, 6-hourly.

    Since the data is already z-score normalized (mean~0, std~1 per channel),
    set data_mean=0 and data_std=1 so that the algorithm's ``_normalize_x``
    becomes identity.  ``clip_noise`` should be raised (e.g. 15) to accommodate
    the ~[-6, 6] data range.
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.n_frames = cfg.n_frames if split == "training" else cfg.n_frames * cfg.validation_multiplier

        subfolder = "train" if split == "training" else "val"
        data_path = Path(cfg.data_dir) / subfolder / "data.pt"
        self.data = torch.load(data_path, map_location="cpu")  # (B, T, C, H, W)

        # Auto-crop spatial dims to multiples of 8 (required for 3 U-Net downsample/upsample stages)
        _, _, _, H, W = self.data.shape
        H_crop = (H // 8) * 8
        W_crop = (W // 8) * 8
        if H_crop != H or W_crop != W:
            self.data = self.data[:, :, :, :H_crop, :W_crop]

        self.num_trajectories = self.data.shape[0]
        self.total_frames = self.data.shape[1]
        self.clips_per_traj = max(1, self.total_frames - self.n_frames + 1)

        # Optional: replicate clip 0 of each trajectory M times for ensemble
        # forecasting (shared starting condition, M independent diffusion samples).
        # Only active for non-training splits and when explicitly enabled.
        self.repeat_for_ensemble = int(cfg.get("repeat_for_ensemble", 1))
        if self.repeat_for_ensemble > 1 and split != "training":
            self.clips_per_traj = self.repeat_for_ensemble
            self._clip0_only = True
        else:
            self._clip0_only = False

    def __len__(self):
        return self.num_trajectories * self.clips_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.clips_per_traj
        frame_idx = 0 if self._clip0_only else (idx % self.clips_per_traj)

        video = self.data[traj_idx, frame_idx : frame_idx + self.n_frames]  # (n_frames, C, H, W)

        pad_len = self.n_frames - len(video)
        nonterminal = np.ones(self.n_frames, dtype=np.float32)

        if pad_len > 0:
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len))
            nonterminal[-pad_len:] = 0

        video = video.float()
        nonterminal = torch.from_numpy(nonterminal)

        return video, nonterminal
