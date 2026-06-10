# Counterfactual Pipeline Runbook

## Goal

Run the notebook research pipeline locally, starting from raw market and macro downloads and ending with generated processed data, trained DDPM checkpoints, evaluation tables, and live counterfactual notebook outputs.

The generated artifacts are intentionally ignored by git. See [ARTIFACTS.md](ARTIFACTS.md).

## Core Files

Configuration:

- `dataset_spec_v1.json`
- `scenario_definitions_v1.csv`
- `ablation_matrix_v1.csv`

Execution helpers:

- `pipeline_preflight_check.py`
- `run_counterfactual_pipeline.py`
- `requirements.txt`
- `counterfactual_pipeline_requirements.txt`

Current notebook order:

1. `release_aware_macro_loader.ipynb`
2. `market_state_and_alignment_builder.ipynb`
3. `training_prep_for_upgraded_ddpm.ipynb`
4. `retrain_upgraded_ddpm.ipynb`
5. `evaluate_upgraded_ddpm.ipynb`
6. `live_counterfactual_lab_v1.ipynb`

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Optional but recommended:

```bash
export FRED_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:FRED_API_KEY="your_key_here"
```

If `FRED_API_KEY` is not set, the macro notebook should use fallback release-lag logic instead of richer vintage-aware access.

## Preflight

Run:

```bash
python pipeline_preflight_check.py
```

Skip network checks when offline:

```bash
python pipeline_preflight_check.py --skip-network
```

Note: the preflight script creates expected output directories under `counterfactual_data_build/` and `executed_notebooks/`.

## Full Run

Run the full notebook sequence:

```bash
python run_counterfactual_pipeline.py
```

Executed notebook copies are written to:

- `executed_notebooks/`

Generated pipeline outputs are written under:

- `counterfactual_data_build/processed/`
- `counterfactual_data_build/outputs/checkpoints/`
- `counterfactual_data_build/outputs/generated_samples/`
- `counterfactual_data_build/outputs/tables/`
- `counterfactual_data_build/outputs/figures/`
- `counterfactual_data_build/outputs/live_counterfactuals/`

## Partial Runs

Start at a specific notebook:

```bash
python run_counterfactual_pipeline.py --start-at training_prep_for_upgraded_ddpm.ipynb
```

Stop after a specific notebook:

```bash
python run_counterfactual_pipeline.py --stop-after retrain_upgraded_ddpm.ipynb
```

## Artifact Dependencies

Later notebooks require artifacts from earlier notebooks:

- `training_prep_for_upgraded_ddpm.ipynb` requires the aligned dataset from `market_state_and_alignment_builder.ipynb`.
- `retrain_upgraded_ddpm.ipynb` requires window arrays and scalers from `training_prep_for_upgraded_ddpm.ipynb`.
- `evaluate_upgraded_ddpm.ipynb` requires the trained checkpoint from `retrain_upgraded_ddpm.ipynb`.
- `live_counterfactual_lab_v1.ipynb` requires processed windows, scalers, model config, and the trained checkpoint.

If those artifacts are missing, run the earlier notebooks first.

## Common First-Run Issues

### `FRED_API_KEY` missing

This is not fatal. The macro notebook should fall back to release-lag approximation mode.

### Yahoo or FRED access errors

These are usually network, firewall, VPN, or temporary provider issues. Re-run the failing notebook after checking connectivity.

### Long retraining time

The retraining notebook is the heaviest step. For a smoke test, reduce runtime settings in the notebook before running a full experiment.

### Notebook execution dependency errors

Install the notebook runtime dependencies:

```bash
pip install nbformat nbclient jupyter ipykernel
```

## Recommended First Run Strategy

1. Run `python pipeline_preflight_check.py --skip-network` to check local packages and files.
2. Run `python pipeline_preflight_check.py` when network access is available.
3. Run through `training_prep_for_upgraded_ddpm.ipynb`.
4. Inspect `counterfactual_data_build/processed/` and `counterfactual_data_build/outputs/tables/window_array_shapes.csv`.
5. Run `retrain_upgraded_ddpm.ipynb`.
6. Run `evaluate_upgraded_ddpm.ipynb`.
7. Run `live_counterfactual_lab_v1.ipynb` only after the upgraded checkpoint exists.
