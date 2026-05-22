import torch
import numpy as np
from omegaconf import DictConfig
from pathlib import Path


class SEVIRDataset(torch.utils.data.Dataset):
    """
    SEVIR-LR VIL dataset for diffusion forcing.

    Expects ``cfg.data_dir`` to point to a folder containing ``train/`` and
    ``val/`` subfolders, each holding a ``data.npy`` array of shape
    ``(B, T, H, W)`` with float32 values in [0, 1].

    Uses numpy memory-mapped loading (``mmap_mode='r'``) so that:
      - Data is lazily paged in from disk, not fully loaded into RAM.
      - With DDP, all processes share the same OS page cache, so 4 GPUs
        use ~the same RAM as 1 GPU (only the active working set is resident).

    Since the data is already in [0, 1], setting data_min=0 and data_max=1
    makes the minmax normalization an identity.  The algorithm's
    ``_normalize_x`` (data_mean=0.5, data_std=0.5) then maps [0, 1] → [-1, 1].
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.n_frames = cfg.n_frames if split == "training" else cfg.n_frames * cfg.validation_multiplier

        subfolder = "train" if split == "training" else "val"
        npy_path = Path(cfg.data_dir) / subfolder / "data.npy"
        self.data = np.load(str(npy_path), mmap_mode="r")  # (B, T, H, W), lazy

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

        # .copy() is required: memmap slices are read-only and non-contiguous
        chunk = self.data[traj_idx, frame_idx : frame_idx + self.n_frames].copy()
        video = torch.from_numpy(chunk)  # (n_frames, H, W)

        pad_len = self.n_frames - len(video)
        nonterminal = np.ones(self.n_frames, dtype=np.float32)

        if pad_len > 0:
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, pad_len))
            nonterminal[-pad_len:] = 0

        # (n_frames, H, W) -> (n_frames, 1, H, W)
        video = video.unsqueeze(1).float()
        nonterminal = torch.from_numpy(nonterminal)

        return video, nonterminal
