"""
Forward measurement operators for inverse problems / data assimilation.

Each operator implements A: x -> y_clean, where y = A(x) + noise.
Operators accept tensors of arbitrary leading dimensions (..., C, H, W).
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ForwardOperator(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SuperResolution(ForwardOperator):
    """Downsample by a given integer factor using bilinear interpolation."""

    def __init__(self, scale_factor: int):
        self.scale_factor = scale_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, *shape[-3:])
        out = F.interpolate(
            x_flat,
            scale_factor=1.0 / self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        return out.reshape(*shape[:-2], *out.shape[-2:])

    def __repr__(self):
        return f"SuperResolution(scale_factor={self.scale_factor})"


class SparseObservation(ForwardOperator):
    """Keep a random subset of pixels (binary mask), zero out the rest."""

    def __init__(self, ratio: float, H: int, W: int, seed: int = 42):
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.ratio = ratio
        self.mask = (torch.rand(1, 1, H, W, generator=gen) < ratio).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask.to(x.device)

    def __repr__(self):
        return f"SparseObservation(ratio={self.ratio})"


class Identity(ForwardOperator):
    """Identity operator (denoising-only inverse problem)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __repr__(self):
        return "Identity()"


def get_operator(cfg) -> ForwardOperator:
    """Instantiate an operator from a Hydra/OmegaConf config node."""
    name = cfg.get("name", "identity")
    if name == "super_resolution":
        return SuperResolution(scale_factor=int(cfg.get("scale_factor", 4)))
    elif name == "sparse_observation":
        return SparseObservation(
            ratio=float(cfg.get("ratio", 0.05)),
            H=int(cfg.get("H", 128)),
            W=int(cfg.get("W", 128)),
            seed=int(cfg.get("seed", 42)),
        )
    elif name == "identity":
        return Identity()
    else:
        raise ValueError(f"Unknown operator: {name}")
