#!/bin/bash
# =============================================================================
# Train ForcingDAS on ERA5 (multi-variable global weather: z500, t850, u10, v10).
#
# Edit the hyperparameters below (or override via environment variables), then:
#     bash scripts/train_era5.sh
#
# Scripts use the DiT backbone (df_era5_dit) by default. For the 3D U-Net
# backbone, set ALGO=df_era5 and delete the DiT architecture overrides below.
# =============================================================================
set -e
cd "$(dirname "$0")/.."
# conda activate forcingdas

# ── Paths / hardware ─────────────────────────────────────────────
DATA_DIR=${DATA_DIR:-/path/to/era5}        # folder with train/, val/, test/ (each data.pt) and stats.pt
GPUS=${GPUS:-0}

# ── Backbone (S 384/12/6 · B 768/12/12 · L 1024/24/16 · XL 1152/28/16) ──
ALGO=${ALGO:-df_era5_dit}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
DEPTH=${DEPTH:-12}
NUM_HEADS=${NUM_HEADS:-12}
PATCH_SIZE=${PATCH_SIZE:-4}
GRAD_CKPT=${GRAD_CKPT:-True}

# ── Data (ERA5 is already z-score normalized per channel) ────────
N_FRAMES=${N_FRAMES:-30}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4}

# ── Optimization ─────────────────────────────────────────────────
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-4e-4}
MAX_STEPS=${MAX_STEPS:-200000}
WARMUP_STEPS=${WARMUP_STEPS:-5000}

# ── Training noise schedule ──────────────────────────────────────
NOISE_LEVEL=${NOISE_LEVEL:-random_causal_exact_pyramid}
CAUSAL_RATIO=${CAUSAL_RATIO:-0.25}
CONTEXT_CLEAN_RATIO=${CONTEXT_CLEAN_RATIO:-0.0}
MAX_CONTEXT_FRAMES=${MAX_CONTEXT_FRAMES:-4}
PYRAMID_RATIO=${PYRAMID_RATIO:-0.5}

# ── Logging / checkpointing ──────────────────────────────────────
VAL_EVERY=${VAL_EVERY:-50000}
CKPT_EVERY=${CKPT_EVERY:-2000}
VIS_EVERY=${VIS_EVERY:-500}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-forcingdas-era5}
WANDB_MODE=${WANDB_MODE:-online}
TAG=${TAG:-era5_${NOISE_LEVEL}_cr${CAUSAL_RATIO}_lr${LR}}

CUDA_VISIBLE_DEVICES=$GPUS python -m main \
  +name=$TAG \
  algorithm=$ALGO \
  experiment=exp_era5 \
  dataset=era5 \
  dataset.data_dir=$DATA_DIR \
  dataset.n_frames=$N_FRAMES \
  dataset.context_length=$CONTEXT_LENGTH \
  experiment.training.batch_size=$BATCH_SIZE \
  experiment.training.lr=$LR \
  experiment.training.max_steps=$MAX_STEPS \
  algorithm.warmup_steps=$WARMUP_STEPS \
  algorithm.noise_level=$NOISE_LEVEL \
  algorithm.causal_ratio=$CAUSAL_RATIO \
  algorithm.context_clean_ratio=$CONTEXT_CLEAN_RATIO \
  algorithm.max_context_frames=$MAX_CONTEXT_FRAMES \
  algorithm.pyramid_ratio=$PYRAMID_RATIO \
  algorithm.diffusion.architecture.hidden_size=$HIDDEN_SIZE \
  algorithm.diffusion.architecture.depth=$DEPTH \
  algorithm.diffusion.architecture.num_heads=$NUM_HEADS \
  algorithm.diffusion.architecture.patch_size=$PATCH_SIZE \
  algorithm.diffusion.architecture.use_gradient_checkpointing=$GRAD_CKPT \
  algorithm.train_vis_every=$VIS_EVERY \
  experiment.training.checkpointing.every_n_train_steps=$CKPT_EVERY \
  experiment.validation.val_every_n_step=$VAL_EVERY \
  experiment.training.data.num_workers=$NUM_WORKERS \
  experiment.validation.data.num_workers=$NUM_WORKERS \
  wandb.project=$WANDB_PROJECT \
  wandb.mode=$WANDB_MODE
