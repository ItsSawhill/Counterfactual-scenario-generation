# Results

## Scope

This page only describes results supported by tracked files in `counterfactual_data_build/outputs/`.

The tracked outputs are summaries and figures from prior notebook runs. The processed datasets, NumPy arrays, generated sample arrays, and PyTorch checkpoints used to produce them are not tracked.

## Tracked Output Files

Tracked tables:

- `counterfactual_data_build/outputs/tables/ddpm_upgraded_model_config.json`
- `counterfactual_data_build/outputs/tables/ddpm_upgraded_training_history.csv`
- `counterfactual_data_build/outputs/tables/ddpm_upgraded_training_summary.json`
- `counterfactual_data_build/outputs/tables/scaling_check_summary.csv`
- `counterfactual_data_build/outputs/tables/time_split_summary.csv`
- `counterfactual_data_build/outputs/tables/upgraded_ddpm_corr_distance.csv`
- `counterfactual_data_build/outputs/tables/upgraded_ddpm_vs_gaussian_summary.csv`
- `counterfactual_data_build/outputs/tables/window_array_shapes.csv`

Tracked figures:

- `counterfactual_data_build/outputs/figures/cl_log_return_upgraded_eval.png`
- `counterfactual_data_build/outputs/figures/gc_log_return_upgraded_eval.png`
- `counterfactual_data_build/outputs/figures/qqq_log_return_upgraded_eval.png`
- `counterfactual_data_build/outputs/figures/spy_log_return_upgraded_eval.png`
- `counterfactual_data_build/outputs/figures/zn_log_return_upgraded_eval.png`
- `counterfactual_data_build/outputs/live_counterfactuals/*.png`

## Training Snapshot

From `counterfactual_data_build/outputs/tables/ddpm_upgraded_training_summary.json`:

- device: `cpu`
- epochs completed: `28`
- max epochs allowed: `35`
- best epoch: `20`
- best validation loss: `0.2605`
- final training loss: `0.1181`
- final validation loss: `0.2739`
- train samples: `2715`
- validation samples: `559`

The summary file references checkpoint paths from the machine where the prior notebook run occurred. Those checkpoint files are not tracked in this repository.

## Window Shapes

From `counterfactual_data_build/outputs/tables/window_array_shapes.csv`:

- `X_train`: `(2715, 30, 5)`
- `C_train`: `(2715, 30, 88)`
- `X_val`: `(559, 30, 5)`
- `C_val`: `(559, 30, 88)`
- `X_test`: `(560, 30, 5)`
- `C_test`: `(560, 30, 88)`

These shape summaries are tracked. The corresponding `.npy` arrays are not tracked.

## Evaluation Highlights

From `counterfactual_data_build/outputs/tables/upgraded_ddpm_corr_distance.csv`:

- DDPM mean absolute correlation distance: `0.0380`
- Gaussian baseline mean absolute correlation distance: `0.1414`

From `counterfactual_data_build/outputs/tables/upgraded_ddpm_vs_gaussian_summary.csv`:

- `SPY_log_return` shows lower DDPM mean absolute error and much lower skew/kurtosis error than the Gaussian baseline.
- `GC_log_return` shows DDPM improvements across several distribution and tail metrics.
- `CL_log_return` remains difficult; DDPM mean and volatility errors are worse than the Gaussian baseline in the tracked table, while some tail and shape metrics improve.
- `ZN_log_return` is mixed; some moments favor the Gaussian baseline and others favor DDPM.

## Missing Scenario Summary

Earlier documentation referenced `live_counterfactual_summary.csv`. That file is not tracked in this repository. The only tracked live counterfactual outputs are PNG figures under `counterfactual_data_build/outputs/live_counterfactuals/`.

Do not cite scenario-summary numbers unless the corresponding CSV is regenerated or added as a documented artifact.

## Interpretation

The tracked results support the claim that a prior notebook run trained and evaluated an upgraded conditional DDPM. They do not prove that the current FastAPI backend serves that trained model, because the backend currently uses fallback simulation.
