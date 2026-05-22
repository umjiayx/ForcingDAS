#!/bin/bash
# =============================================================================
# Data assimilation on ERA5 with a trained ForcingDAS model.
#
# Edit the hyperparameters below (or override via environment variables), then:
#     bash scripts/test_era5.sh
#
# The DA regime is set by (SCHEDULING_MATRIX, CHUNK_SIZE):
#     filtering / nowcasting   ->  autoregressive , 1
#     fixed-lag smoothing      ->  pyramid        , -1   (default)
#     batch reanalysis         ->  full_sequence  , -1
# The backbone settings must match the checkpoint.
# ERA5 fields are 4-channel at 240x120; the sparse-observation mask uses OP_H/OP_W.
# =============================================================================
set -e
cd "$(dirname "$0")/.."
# conda activate forcingdas

# ── Paths / hardware ─────────────────────────────────────────────
DATA_DIR=${DATA_DIR:-/path/to/era5}
CKPT_PATH=${CKPT_PATH:-/path/to/checkpoint.ckpt}
GPU=${GPU:-0}

# ── Backbone (must match the checkpoint) ─────────────────────────
ALGO=${ALGO:-df_era5_dit}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
DEPTH=${DEPTH:-12}
NUM_HEADS=${NUM_HEADS:-12}
PATCH_SIZE=${PATCH_SIZE:-4}

# ── Data ─────────────────────────────────────────────────────────
N_FRAMES=${N_FRAMES:-30}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4}

# ── DA regime + sampling ─────────────────────────────────────────
SCHEDULING_MATRIX=${SCHEDULING_MATRIX:-pyramid}   # autoregressive | pyramid | full_sequence
CHUNK_SIZE=${CHUNK_SIZE:--1}
UNCERTAINTY_SCALE=${UNCERTAINTY_SCALE:-1}
DDIM_STEPS=${DDIM_STEPS:-100}

# ── Observation guidance ─────────────────────────────────────────
OPERATOR=${OPERATOR:-sparse_observation}   # identity | super_resolution | sparse_observation
RATIO=${RATIO:-0.10}
SCALE_FACTOR=${SCALE_FACTOR:-4}
OP_H=${OP_H:-240}                           # ERA5 spatial height
OP_W=${OP_W:-120}                           # ERA5 spatial width
NOISE_SIGMA=${NOISE_SIGMA:-0.05}
GRAD_SCALE=${GRAD_SCALE:-2.0}
GAMMA=${GAMMA:-0.01}

# ── Evaluation / logging ─────────────────────────────────────────
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-1}
LIMIT_BATCH=${LIMIT_BATCH:-4}               # null to run the full test set
WANDB_PROJECT=${WANDB_PROJECT:-forcingdas-era5-da}
WANDB_MODE=${WANDB_MODE:-online}
TAG=${TAG:-era5_${SCHEDULING_MATRIX}_${N_FRAMES}f_ctx${CONTEXT_LENGTH}_g${GAMMA}_gs${GRAD_SCALE}}

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -m main \
  +name=$TAG \
  "+load='$CKPT_PATH'" \
  algorithm=$ALGO \
  experiment=exp_era5 \
  dataset=era5 \
  dataset.data_dir=$DATA_DIR \
  dataset.n_frames=$N_FRAMES \
  dataset.context_length=$CONTEXT_LENGTH \
  experiment.tasks=[test] \
  experiment.test.batch_size=$TEST_BATCH_SIZE \
  experiment.test.limit_batch=$LIMIT_BATCH \
  algorithm.scheduling_matrix=$SCHEDULING_MATRIX \
  algorithm.chunk_size=$CHUNK_SIZE \
  algorithm.uncertainty_scale=$UNCERTAINTY_SCALE \
  algorithm.diffusion.sampling_timesteps=$DDIM_STEPS \
  algorithm.diffusion.architecture.hidden_size=$HIDDEN_SIZE \
  algorithm.diffusion.architecture.depth=$DEPTH \
  algorithm.diffusion.architecture.num_heads=$NUM_HEADS \
  algorithm.diffusion.architecture.patch_size=$PATCH_SIZE \
  algorithm.obs_guidance.enabled=true \
  algorithm.obs_guidance.operator.name=$OPERATOR \
  algorithm.obs_guidance.operator.ratio=$RATIO \
  algorithm.obs_guidance.operator.scale_factor=$SCALE_FACTOR \
  algorithm.obs_guidance.operator.H=$OP_H \
  algorithm.obs_guidance.operator.W=$OP_W \
  algorithm.obs_guidance.noise_sigma=$NOISE_SIGMA \
  algorithm.obs_guidance.grad_scale=$GRAD_SCALE \
  algorithm.obs_guidance.gamma=$GAMMA \
  wandb.project=$WANDB_PROJECT \
  wandb.mode=$WANDB_MODE
