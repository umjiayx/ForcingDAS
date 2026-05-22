"""
DiT (Diffusion Transformer) backbone for Diffusion Forcing.

Drop-in replacement for Unet3D with the same forward interface:
    forward(x, noise_levels, external_cond, is_causal)
    where x: (B, C, F, H, W), noise_levels: (F, B)

Supports block-causal temporal attention for autoregressive generation
with pyramid scheduling matrix.

Two attention modes:
  - "factorized": spatial attn (within-frame) + causal temporal attn (across-frames).
    Memory-efficient, uses Flash Attention for both parts. Recommended for
    high-resolution inputs (e.g. ERA5 grids).
  - "full": full space-time attention with block-causal mask.
    Better modeling capacity but O((T*P)^2) cost. Only practical for small P.
"""

from typing import Optional, Literal
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .embeddings import Timesteps, TimestepEmbedding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Embed 2D image patches via Conv2d. Supports non-square inputs."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        """(B, C, H, W) -> (B, H'*W', D), h_patches, w_patches"""
        x = self.proj(x)
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = rearrange(x, "b d h w -> b (h w) d")
        return x, h_patches, w_patches


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero: returns (modulated_x, gate)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return _modulate(self.norm(x), shift, scale), gate


class AdaLN(nn.Module):
    """Adaptive LayerNorm (no gate)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return _modulate(self.norm(x), shift, scale)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.qkv.weight)
        if qkv_bias:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# DiT Blocks
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Standard DiT block with full attention and AdaLN-Zero."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = AdaLNZero(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = AdaLNZero(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        h, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(h, attn_mask=attn_mask, is_causal=is_causal)
        h, gate_mlp = self.norm2(x, c)
        x = x + gate_mlp * self.mlp(h)
        return x


class FactorizedDiTBlock(nn.Module):
    """
    Factorized DiT block: spatial attention + temporal attention + MLP.

    Spatial attention operates within each frame (all patches attend to each other).
    Temporal attention operates across frames (each spatial position attends across time).
    This decomposition is much more memory-efficient for video and allows using
    Flash Attention (is_causal=True) for the temporal component.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.spatial_norm = AdaLNZero(hidden_size)
        self.spatial_attn = Attention(hidden_size, num_heads=num_heads)
        self.temporal_norm = AdaLNZero(hidden_size)
        self.temporal_attn = Attention(hidden_size, num_heads=num_heads)
        self.mlp_norm = AdaLNZero(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        num_frames: int,
        num_patches: int,
        is_causal: bool = False,
    ):
        B = x.shape[0]

        # --- Spatial self-attention (within each frame, no causal mask) ---
        x_s = rearrange(x, "b (f p) d -> (b f) p d", f=num_frames, p=num_patches)
        c_s = rearrange(c, "b (f p) d -> (b f) p d", f=num_frames, p=num_patches)
        h, gate = self.spatial_norm(x_s, c_s)
        x_s = x_s + gate * self.spatial_attn(h)
        x = rearrange(x_s, "(b f) p d -> b (f p) d", b=B)

        # --- Temporal self-attention (across frames, causal if requested) ---
        x_t = rearrange(x, "b (f p) d -> (b p) f d", f=num_frames, p=num_patches)
        c_t = rearrange(c, "b (f p) d -> (b p) f d", f=num_frames, p=num_patches)
        h, gate = self.temporal_norm(x_t, c_t)
        x_t = x_t + gate * self.temporal_attn(h, is_causal=is_causal)
        x = rearrange(x_t, "(b p) f d -> b (f p) d", b=B, p=num_patches)

        # --- MLP ---
        h, gate = self.mlp_norm(x, c)
        x = x + gate * self.mlp(h)
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = AdaLN(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        return self.linear(self.norm(x, c))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DiT3DVideo(nn.Module):
    """
    DiT backbone for video diffusion forcing.

    Drop-in replacement for Unet3D — same forward signature::

        output = model(x, noise_levels, external_cond, is_causal)
        # x:            (B, C, F, H, W)
        # noise_levels:  (F, B)
        # output:        (B, C, F, H, W)

    Supports block-causal temporal attention for autoregressive generation
    with pyramid / trapezoid scheduling matrices.

    Named sizes (following ViT conventions)::

        DiT-S:  ~33M params  (hidden=384,  depth=12, heads=6)
        DiT-B: ~130M params  (hidden=768,  depth=12, heads=12)
        DiT-L: ~458M params  (hidden=1024, depth=24, heads=16)
        DiT-XL:~675M params  (hidden=1152, depth=28, heads=16)

    Args:
        channels: Number of input channels per frame.
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        patch_size: Spatial patch size for tokenization.
        mlp_ratio: MLP hidden dim / hidden_size.
        max_frames: Maximum temporal length (for positional embedding).
        max_spatial_tokens: Maximum H'*W' after patching (for positional embedding).
            Set to ceil(H/patch_size) * ceil(W/patch_size) for your dataset.
        is_causal: Default causal mode. Can be overridden per forward call.
        attention_type: "factorized" (recommended) or "full".
        use_gradient_checkpointing: Trade compute for memory in transformer blocks.
        external_cond_dim: (unused, for interface compat with Unet3D).
    """

    SIZES = {
        "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
        "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
        "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
        "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
    }

    def __init__(
        self,
        channels: int,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        max_frames: int = 64,
        max_spatial_tokens: int = 1024,
        is_causal: bool = True,
        attention_type: Literal["full", "factorized"] = "factorized",
        use_gradient_checkpointing: bool = False,
        external_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        if external_cond_dim:
            raise NotImplementedError(
                "External conditioning not yet implemented for DiT3DVideo"
            )

        self.channels = channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.is_causal = is_causal
        self.attention_type = attention_type
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # --- Patch embedding ---
        self.patch_embed = PatchEmbed(channels, hidden_size, patch_size)

        # --- Learnable positional embedding (large enough for max sequence) ---
        max_tokens = max_frames * max_spatial_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Per-frame noise level embedding ---
        noise_level_dim = max(hidden_size // 4, 64)
        self.noise_level_embedding = nn.Sequential(
            Timesteps(noise_level_dim, True, 0),
            TimestepEmbedding(
                in_channels=noise_level_dim, time_embed_dim=hidden_size,
            ),
        )

        # --- Transformer blocks ---
        block_cls = (
            FactorizedDiTBlock
            if attention_type == "factorized"
            else DiTBlock
        )
        self.blocks = nn.ModuleList(
            [block_cls(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # --- Output projection ---
        out_channels = patch_size**2 * channels
        self.final_layer = DiTFinalLayer(hidden_size, out_channels)

        # --- Block-causal mask cache (for "full" attention mode) ---
        self._causal_mask_cache = {}

    def _get_block_causal_mask(
        self, num_frames: int, num_patches: int, device: torch.device,
    ) -> torch.Tensor:
        """
        Build block-causal mask for full space-time attention.
        Frame t can attend to all patches of frames <= t.
        Returns a bool mask where True = allowed to attend.
        """
        key = (num_frames, num_patches, device)
        if key not in self._causal_mask_cache:
            N = num_frames * num_patches
            frame_idx = torch.arange(N, device=device) // num_patches
            mask = frame_idx.unsqueeze(0) >= frame_idx.unsqueeze(1)
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def _run_block(self, block, *args, **kwargs):
        if self.use_gradient_checkpointing and (self.training or torch.is_grad_enabled()):
            return torch.utils.checkpoint.checkpoint(
                block, *args, use_reentrant=False, **kwargs,
            )
        return block(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, F, H, W) video tensor (after EinopsWrapper rearranges from f b c h w).
            noise_levels: (F, B) per-frame noise level indices.
            external_cond: unused, kept for interface compatibility.
            is_causal: override default causal mode.
        Returns:
            (B, C, F, H, W) predicted output.
        """
        use_causal = is_causal if is_causal is not None else self.is_causal
        B, C, F, H, W = x.shape

        # 1. Patchify each frame
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x, h_patches, w_patches = self.patch_embed(x)
        P = h_patches * w_patches
        x = rearrange(x, "(b f) p d -> b (f p) d", b=B)

        # 2. Positional embedding
        seq_len = F * P
        x = x + self.pos_embed[:, :seq_len]

        # 3. Noise level embedding: (F, B) -> (B, F) -> (B, F, D) -> (B, F*P, D)
        noise_levels_bf = rearrange(noise_levels, "f b -> b f")
        emb = self.noise_level_embedding(noise_levels_bf)
        emb = repeat(emb, "b f d -> b (f p) d", p=P)

        # 4. Transformer blocks
        if self.attention_type == "factorized":
            for block in self.blocks:
                x = self._run_block(
                    block, x, emb,
                    num_frames=F, num_patches=P, is_causal=use_causal,
                )
        else:
            attn_mask = (
                self._get_block_causal_mask(F, P, x.device)
                if use_causal
                else None
            )
            for block in self.blocks:
                x = self._run_block(block, x, emb, attn_mask=attn_mask)

        # 5. Final projection
        x = self.final_layer(x, emb)

        # 6. Unpatchify
        x = rearrange(x, "b (f p) d -> (b f) p d", f=F)
        x = rearrange(
            x,
            "bf (h w) (ph pw c) -> bf c (h ph) (w pw)",
            h=h_patches, w=w_patches,
            ph=self.patch_size, pw=self.patch_size,
        )
        x = rearrange(x, "(b f) c h w -> b c f h w", b=B)

        return x
