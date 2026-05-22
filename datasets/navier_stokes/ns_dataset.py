import torch
import numpy as np
from omegaconf import DictConfig
from pathlib import Path


class NavierStokesDataset(torch.utils.data.Dataset):
    """
    Navier-Stokes vorticity dataset.

    Expects ``cfg.data_dir`` to point to a folder containing ``train/`` and
    ``val/`` subfolders, each holding a ``data.pt`` tensor of shape
    ``(B, T, H, W)``.

    Normalization (``cfg.normalization``):

    - ``"minmax"`` (default): min-max to [0, 1] using global data_min / data_max.
      The algorithm's ``_normalize_x`` (data_mean=0.5, data_std=0.5) then maps
      [0, 1] → [-1, 1].

    - ``"zscore"``: divides raw vorticity by ``data_raw_std``.
      Result has mean ≈ 0, std ≈ 1, range ≈ [-6, 6].  With data_mean=0,
      data_std=1, ``_normalize_x`` becomes identity.
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.normalization = cfg.get("normalization", "minmax")

        self.n_frames = cfg.n_frames if split == "training" else cfg.n_frames * cfg.validation_multiplier

        subfolder = "train" if split == "training" else "val"
        data_path = Path(cfg.data_dir) / subfolder / "data.pt"
        self.data = torch.load(data_path, map_location="cpu")  # (B, T, H, W)

        if self.normalization == "zscore":
            self.data = self.data / cfg.data_raw_std
        else:
            self.data = (self.data - cfg.data_min) / (cfg.data_max - cfg.data_min)

        # Optional trajectory-axis slice for non-training splits. Used to fan
        # out a multi-trajectory test/validation run across multiple GPUs by
        # launching one process per slice (start_idx, n_traj) and aggregating
        # the saved predictions afterwards. Training always sees the full set.
        start_idx = int(cfg.get("start_idx", 0))
        n_traj_cfg = cfg.get("n_traj", None)
        if split != "training" and (start_idx != 0 or n_traj_cfg is not None):
            full_n = self.data.shape[0]
            if not (0 <= start_idx < full_n):
                raise ValueError(
                    f"NavierStokesDataset[{split}]: start_idx={start_idx} out of range "
                    f"for dataset with {full_n} trajectories"
                )
            end = full_n if n_traj_cfg is None else min(full_n, start_idx + int(n_traj_cfg))
            self.data = self.data[start_idx:end]

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

        video = self.data[traj_idx, frame_idx : frame_idx + self.n_frames]  # (n_frames, H, W)

        pad_len = self.n_frames - len(video)
        nonterminal = np.ones(self.n_frames, dtype=np.float32)

        if pad_len > 0:
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, pad_len))
            nonterminal[-pad_len:] = 0

        # (n_frames, H, W) -> (n_frames, 1, H, W)
        video = video.unsqueeze(1).float()
        nonterminal = torch.from_numpy(nonterminal)

        return video, nonterminal
