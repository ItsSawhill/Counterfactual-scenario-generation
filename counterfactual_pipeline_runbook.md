# Counterfactual Pipeline Runbook

## Goal

Run the upgraded end-to-end pipeline locally, starting from raw market and macro downloads and ending with:

- trained upgraded DDPM checkpoints
- evaluation tables and figures
- a live counterfactual scenario lab

## Files

Core config:

- [dataset_spec_v1.json](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/dataset_spec_v1.json)
- [scenario_definitions_v1.csv](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/scenario_definitions_v1.csv)
- [ablation_matrix_v1.csv](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/ablation_matrix_v1.csv)

Execution helpers:

- [pipeline_preflight_check.py](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/pipeline_preflight_check.py)
- [run_counterfactual_pipeline.py](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/run_counterfactual_pipeline.py)
- [counterfactual_pipeline_requirements.txt](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/counterfactual_pipeline_requirements.txt)

Notebook order:

1. [release_aware_macro_loader.ipynb](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/release_aware_macro_loader.ipynb)
2. [market_state_and_alignment_builder.ipynb](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/market_state_and_alignment_builder.ipynb)
3. [training_prep_for_upgraded_ddpm.ipynb](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/training_prep_for_upgraded_ddpm.ipynb)
4. [retrain_upgraded_ddpm.ipynb](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/retrain_upgraded_ddpm.ipynb)
5. [evaluate_upgraded_ddpm.ipynb](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/evaluate_upgraded_ddpm.ipynb)
6. [live_counterfactual_lab_v1.ipynb](C:/Users/sammy/Documents/Codex/2026-04-21-files-mentioned-by-the-user-chartpanel/live_counterfactual_lab_v1.ipynb)

## Setup

From this workspace:

```powershell
cd C:\Users\sammy\Documents\Codex\2026-04-21-files-mentioned-by-the-user-chartpanel
pip install -r counterfactual_pipeline_requirements.txt
```

Optional but recommended:

```powershell
$env:FRED_API_KEY="your_key_here"
```

If `FRED_API_KEY` is not set, the macro notebook still runs, but it uses fallback release-lag logic instead of richer vintage-aware access.

## Preflight

Run:

```powershell
python pipeline_preflight_check.py
```

If you already know your machine has no external network access and you just want to validate local setup, use:

```powershell
python pipeline_preflight_check.py --skip-network
```

## Full Run

Run the whole notebook sequence:

```powershell
python run_counterfactual_pipeline.py
```

Executed notebook copies will be saved under:

- `executed_notebooks`

Generated pipeline outputs will be saved under:

- `counterfactual_data_build`

## Partial Runs

Start partway through:

```powershell
python run_counterfactual_pipeline.py --start-at training_prep_for_upgraded_ddpm.ipynb
```

Stop after retraining:

```powershell
python run_counterfactual_pipeline.py --stop-after retrain_upgraded_ddpm.ipynb
```

## Common First-Run Issues

### `FRED_API_KEY` missing

This is not fatal. You lose vintage-aware behavior, but the pipeline should still run in fallback mode.

### Yahoo or FRED access errors

These are usually network, firewall, VPN, or temporary provider issues. Re-run the failing notebook after checking connectivity.

### Long retraining time

The retraining notebook is the heaviest step. If you want a quicker smoke test first, lower:

- `max_epochs`
- `sample_preview_windows`
- `generated_paths`

### Notebook execution dependency errors

Install:

```powershell
pip install nbformat nbclient jupyter ipykernel
```

## Recommended First Run Strategy

1. Run `pipeline_preflight_check.py`
2. Run the first three notebooks only
3. Inspect the aligned dataset and window shapes
4. Run retraining overnight or with a reduced epoch count first
5. Run evaluation
6. Run the live counterfactual lab after the upgraded checkpoint exists
