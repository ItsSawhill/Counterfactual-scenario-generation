# Results

## Current Tracked Outputs

The repository already includes representative generated artifacts under `counterfactual_data_build/outputs/`, including:

- evaluation figures for `SPY`, `QQQ`, `GC`, `CL`, and `ZN`
- training history and model configuration tables
- scenario summary outputs for multiple counterfactual regimes

## Training Snapshot

From `ddpm_upgraded_training_summary.json`:

- device: `cpu`
- epochs completed: `28`
- best epoch: `20`
- best validation loss: `0.2605`
- final training loss: `0.1181`
- final validation loss: `0.2739`

## Evaluation Highlights

From `upgraded_ddpm_corr_distance.csv`:

- DDPM mean absolute correlation distance: `0.0380`
- Gaussian baseline mean absolute correlation distance: `0.1414`

This suggests the upgraded DDPM preserves cross-asset dependency structure materially better than the Gaussian reference baseline.

From `upgraded_ddpm_vs_gaussian_summary.csv`:

- `SPY` shows lower DDPM mean absolute error and much lower skew/kurtosis error than the Gaussian baseline.
- `GC` shows DDPM improvements across mean error, volatility error, tail quantiles, and Wasserstein distance.
- `CL` remains the most difficult asset, with heavy tail behavior still challenging for the current model.
- `ZN` shows mixed results, indicating some moments are competitive while others still favor the Gaussian baseline.

## Scenario Summary Snapshot

From `live_counterfactual_summary.csv`:

- Baseline `SPY` median terminal return is approximately `7.67%`.
- Baseline `CL` median terminal return is strongly negative, around `-29.08%`.
- The `growth_scare` scenario increases `GC` median terminal return to roughly `6.10%`, consistent with a defensive tilt.
- The `hawkish_policy` and `inflation_reacceleration` scenarios leave `CL` deeply negative in the tracked output set, signaling persistent downside pressure in the generated paths.

## Suggested Presentation Use

For a recruiter-facing demo, the strongest story is:

1. The pipeline is end-to-end, from release-aware macro ingestion to scenario generation.
2. The model is evaluated against a simpler statistical baseline rather than presented without controls.
3. The project includes both research artifacts and a frontend interpretation layer.

## Recommended Next Improvements

- Re-run evaluation with multiple seeds and summarize variance bands.
- Save small, presentation-ready figures under `outputs/figures/` for polished demos.
- Export a concise benchmark table for README badges or portfolio screenshots.
