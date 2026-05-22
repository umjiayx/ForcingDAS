# scripts

Launch scripts and utilities. **Each script exposes all hyperparameters as variables at the top of
the file** — edit them in place, or override any as an environment variable
(e.g. `LR=1e-4 BATCH_SIZE=4 bash scripts/train_ns.sh`). Replace the `/path/to/...` and `CKPT_PATH`
placeholders before running.

## Training
- `train_ns.sh`, `train_sevir.sh`, `train_era5.sh` — train ForcingDAS on each domain.
  Knobs include the DiT backbone size (hidden/depth/heads/patch), batch size, learning rate,
  number of steps, and the training noise schedule. Scripts default to the DiT backbone
  (`df_*_dit`); set `ALGO=df_ns` (etc.) and drop the DiT architecture flags for the 3D U-Net.

## Data assimilation (inference)
- `test_ns.sh`, `test_sevir.sh`, `test_era5.sh` — run DA with a trained checkpoint.
  Select the DA regime with `(SCHEDULING_MATRIX, CHUNK_SIZE)`:
  `autoregressive,1` (filtering / nowcasting), `pyramid,-1` (fixed-lag smoothing),
  or `full_sequence,-1` (batch reanalysis). The same checkpoint serves all three regimes.
  The observation operator and its noise/guidance settings are also set at the top of each script.

## Utilities
- `compute_reference_spectrum.py` — precompute a reference power spectrum (`.pt`) for the optional
  spectral regularizer; pass its path via `algorithm.obs_guidance.spectral_ref=...`.
- `compute_forecasting_metrics.py` — compute forecasting/assimilation metrics from saved predictions.

All flags can also be overridden directly on the command line via Hydra; see the top-level `README.md`.
