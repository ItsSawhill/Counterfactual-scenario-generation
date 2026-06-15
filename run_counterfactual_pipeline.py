from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXECUTED_DIR = ROOT / "executed_notebooks"
SMOKE_ROOT = ROOT / "counterfactual_data_build" / "smoke"
MLFLOW_TRACKING_DIR = ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "counterfactual-scenario-generation"
SMOKE_MLFLOW_RUN_NAME = "smoke-ddpm"
MAX_MLFLOW_ARRAY_ARTIFACT_BYTES = 10 * 1024 * 1024

NOTEBOOK_SEQUENCE = [
    "release_aware_macro_loader.ipynb",
    "market_state_and_alignment_builder.ipynb",
    "training_prep_for_upgraded_ddpm.ipynb",
    "retrain_upgraded_ddpm.ipynb",
    "evaluate_upgraded_ddpm.ipynb",
    "live_counterfactual_lab_v1.ipynb",
]


def smoke_artifact_paths(root: Path = SMOKE_ROOT) -> dict[str, Path]:
    windows_dir = root / "processed" / "windows"
    outputs_dir = root / "outputs"
    return {
        "root": root,
        "windows_dir": windows_dir,
        "tables_dir": outputs_dir / "tables",
        "checkpoints_dir": outputs_dir / "checkpoints",
        "samples_dir": outputs_dir / "generated_samples",
        "x_train": windows_dir / "X_train_smoke.npy",
        "c_train": windows_dir / "C_train_smoke.npy",
        "x_val": windows_dir / "X_val_smoke.npy",
        "c_val": windows_dir / "C_val_smoke.npy",
        "x_test": windows_dir / "X_test_smoke.npy",
        "c_test": windows_dir / "C_test_smoke.npy",
        "scalers": windows_dir / "scalers_smoke.json",
        "best_checkpoint": outputs_dir / "checkpoints" / "conditional_ddpm_smoke_best.pt",
        "last_checkpoint": outputs_dir / "checkpoints" / "conditional_ddpm_smoke_last.pt",
        "model_config": outputs_dir / "tables" / "ddpm_smoke_model_config.json",
        "training_summary": outputs_dir / "tables" / "ddpm_smoke_training_summary.json",
        "sample_scaled": outputs_dir / "generated_samples" / "ddpm_smoke_generated_scaled.npy",
        "sample_unscaled": outputs_dir / "generated_samples" / "ddpm_smoke_generated_unscaled.npy",
    }


def load_notebook_runtime():
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as exc:
        raise RuntimeError(
            "Notebook execution dependencies are missing. Install them with:\n"
            "pip install nbformat nbclient jupyter ipykernel"
        ) from exc
    return nbformat, NotebookClient


def select_sequence(start_at: str | None, stop_after: str | None) -> list[str]:
    sequence = NOTEBOOK_SEQUENCE[:]

    if start_at:
        if start_at not in sequence:
            raise ValueError(f"--start-at notebook not found in sequence: {start_at}")
        sequence = sequence[sequence.index(start_at):]

    if stop_after:
        if stop_after not in sequence:
            raise ValueError(f"--stop-after notebook not found in active sequence: {stop_after}")
        sequence = sequence[: sequence.index(stop_after) + 1]

    return sequence


def execute_notebook(nbformat, NotebookClient, notebook_path: Path, timeout: int | None, kernel_name: str) -> Path:
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    output_path = EXECUTED_DIR / notebook_path.name
    nbformat.write(notebook, output_path)
    return output_path


def _standardize(train_values, *splits):
    import numpy as np

    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale, tuple((split - mean) / scale for split in splits)


def _build_windows(frame, target_cols, condition_cols, seq_len):
    import numpy as np
    import pandas as pd

    target_values = frame[target_cols].to_numpy(dtype=np.float32)
    condition_values = frame[condition_cols].to_numpy(dtype=np.float32)
    dates = frame["Date"].to_numpy()

    x_windows = []
    c_windows = []
    end_dates = []
    for end_idx in range(seq_len - 1, len(frame)):
        start_idx = end_idx - seq_len + 1
        x_windows.append(target_values[start_idx:end_idx + 1])
        c_windows.append(condition_values[start_idx:end_idx + 1])
        end_dates.append(pd.Timestamp(dates[end_idx]).date().isoformat())

    if not x_windows:
        raise RuntimeError("Smoke run could not create any windows. Increase downloaded history.")

    return np.stack(x_windows).astype(np.float32), np.stack(c_windows).astype(np.float32), end_dates


def _log_artifact_if_exists(mlflow, path: Path, artifact_path: str | None = None) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _log_array_artifact_if_small(mlflow, path: Path, artifact_path: str | None = None) -> None:
    if path.exists() and path.stat().st_size <= MAX_MLFLOW_ARRAY_ARTIFACT_BYTES:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _log_smoke_mlflow_run(
    *,
    paths: dict[str, Path],
    model_config: dict,
    scaler_payload: dict,
    summary: dict,
) -> Path:
    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(
            "MLflow is required for smoke tracking. Install dependencies with `pip install -r requirements.txt` "
            "or rerun smoke mode with `--disable-mlflow`."
        ) from exc

    tracking_dir = MLFLOW_TRACKING_DIR
    tracking_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=SMOKE_MLFLOW_RUN_NAME):
        mlflow.log_params(
            {
                "generator_mode": "smoke_ddpm",
                "asset_universe": "SPY",
                "seq_len": model_config["seq_len"],
                "forecast_horizon": scaler_payload["forecast_horizon"],
                "input_dim": model_config["input_dim"],
                "condition_dim": model_config["condition_dim"],
                "hidden_dim": model_config["hidden_dim"],
                "diffusion_steps": model_config["num_diffusion_steps"],
                "epochs": summary["num_epochs_completed"],
                "checkpoint_path": str(paths["best_checkpoint"]),
                "scalers_path": str(paths["scalers"]),
                "generated_samples_path": str(paths["sample_unscaled"]),
            }
        )
        metrics = {
            "final_train_loss": summary["final_train_loss"],
            "smoke_runtime_seconds": summary["total_seconds"],
            "number_train_windows": summary["train_samples"],
            "number_val_windows": summary["val_samples"],
            "number_test_windows": summary["test_samples"],
        }
        if summary.get("final_val_loss") is not None:
            metrics["final_val_loss"] = summary["final_val_loss"]
        mlflow.log_metrics(metrics)

        _log_artifact_if_exists(mlflow, paths["model_config"], artifact_path="config")
        _log_artifact_if_exists(mlflow, paths["scalers"], artifact_path="config")
        _log_artifact_if_exists(mlflow, paths["best_checkpoint"], artifact_path="checkpoints")
        _log_array_artifact_if_small(mlflow, paths["sample_scaled"], artifact_path="generated_samples")
        _log_array_artifact_if_small(mlflow, paths["sample_unscaled"], artifact_path="generated_samples")
        _log_artifact_if_exists(mlflow, paths["training_summary"], artifact_path="summaries")
        for metadata_name in ("meta_train_smoke.csv", "meta_val_smoke.csv", "meta_test_smoke.csv"):
            _log_artifact_if_exists(mlflow, paths["windows_dir"] / metadata_name, artifact_path="summaries")

    return tracking_dir


def _smoke_timestep_embedding(timesteps, dim):
    import math
    import torch
    import torch.nn.functional as F

    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _run_smoke_pipeline(enable_mlflow: bool = True) -> int:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yfinance as yf

    start_time = time.time()
    paths = smoke_artifact_paths()
    for key in ("windows_dir", "tables_dir", "checkpoints_dir", "samples_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    print("Running smoke validation pipeline.")
    print("Smoke artifacts root:", paths["root"])
    print("Mode: one asset (SPY), approximately one year of data, tiny DDPM-style training.")

    raw = yf.download(
        "SPY",
        period="1y",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Smoke run could not download SPY data from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_frame = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
        price = price_frame["SPY"] if "SPY" in price_frame.columns else price_frame.iloc[:, 0]
    else:
        price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(price.index).normalize(),
            "SPY_price": price.astype(float).to_numpy(),
        }
    )
    frame["SPY_log_return"] = np.log(frame["SPY_price"] / frame["SPY_price"].shift(1))
    frame["SPY_return_5d"] = frame["SPY_price"].pct_change(5)
    frame["SPY_realized_vol_5d"] = frame["SPY_log_return"].rolling(5).std(ddof=0)
    frame = frame.dropna().reset_index(drop=True)
    if len(frame) < 80:
        raise RuntimeError(f"Smoke run needs at least 80 rows after feature engineering; got {len(frame)}.")

    target_cols = ["SPY_log_return"]
    condition_cols = ["SPY_price", "SPY_return_5d", "SPY_realized_vol_5d"]
    seq_len = 10
    train_end = int(len(frame) * 0.70)
    val_end = int(len(frame) * 0.85)
    train_df = frame.iloc[:train_end].copy()
    val_df = frame.iloc[train_end:val_end].copy()
    test_df = frame.iloc[val_end:].copy()

    target_mean, target_scale, (train_target, val_target, test_target) = _standardize(
        train_df[target_cols].to_numpy(dtype=np.float64),
        train_df[target_cols].to_numpy(dtype=np.float64),
        val_df[target_cols].to_numpy(dtype=np.float64),
        test_df[target_cols].to_numpy(dtype=np.float64),
    )
    condition_mean, condition_scale, (train_condition, val_condition, test_condition) = _standardize(
        train_df[condition_cols].to_numpy(dtype=np.float64),
        train_df[condition_cols].to_numpy(dtype=np.float64),
        val_df[condition_cols].to_numpy(dtype=np.float64),
        test_df[condition_cols].to_numpy(dtype=np.float64),
    )

    for split_df, target_values, condition_values in [
        (train_df, train_target, train_condition),
        (val_df, val_target, val_condition),
        (test_df, test_target, test_condition),
    ]:
        split_df.loc[:, target_cols] = target_values
        split_df.loc[:, condition_cols] = condition_values

    X_train, C_train, train_dates = _build_windows(train_df, target_cols, condition_cols, seq_len)
    X_val, C_val, val_dates = _build_windows(val_df, target_cols, condition_cols, seq_len)
    X_test, C_test, test_dates = _build_windows(test_df, target_cols, condition_cols, seq_len)

    np.save(paths["x_train"], X_train)
    np.save(paths["c_train"], C_train)
    np.save(paths["x_val"], X_val)
    np.save(paths["c_val"], C_val)
    np.save(paths["x_test"], X_test)
    np.save(paths["c_test"], C_test)
    pd.DataFrame({"split": "train_smoke", "window_end_date": train_dates}).to_csv(paths["windows_dir"] / "meta_train_smoke.csv", index=False)
    pd.DataFrame({"split": "val_smoke", "window_end_date": val_dates}).to_csv(paths["windows_dir"] / "meta_val_smoke.csv", index=False)
    pd.DataFrame({"split": "test_smoke", "window_end_date": test_dates}).to_csv(paths["windows_dir"] / "meta_test_smoke.csv", index=False)

    scaler_payload = {
        "artifact_mode": "smoke",
        "target_cols": target_cols,
        "condition_cols": condition_cols,
        "return_scaler_mean": target_mean.tolist(),
        "return_scaler_scale": target_scale.tolist(),
        "condition_scaler_mean": condition_mean.tolist(),
        "condition_scaler_scale": condition_scale.tolist(),
        "seq_len": seq_len,
        "forecast_horizon": seq_len,
    }
    paths["scalers"].write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")

    class SmokeDenoiser(nn.Module):
        def __init__(self, input_dim, condition_dim, hidden_dim=24, time_dim=24):
            super().__init__()
            self.time_dim = time_dim
            self.input_proj = nn.Linear(input_dim + condition_dim + time_dim, hidden_dim)
            self.hidden = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            self.output = nn.Linear(hidden_dim, input_dim)

        def forward(self, x, c, t):
            t_emb = _smoke_timestep_embedding(t, self.time_dim).unsqueeze(1).repeat(1, x.shape[1], 1)
            h = torch.cat([x, c, t_emb], dim=-1)
            return self.output(self.hidden(self.input_proj(h)))

    class SmokeScheduler:
        def __init__(self, num_steps=20, beta_start=1e-4, beta_end=1.5e-2, device="cpu"):
            self.num_steps = num_steps
            self.device = device
            self.betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
            self.alphas = 1.0 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        def q_sample(self, x0, t, noise):
            sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
            sqrt_omb = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1)
            return sqrt_ab * x0 + sqrt_omb * noise

        def sample_reverse(self, model, cond, shape):
            x = torch.randn(shape, device=self.device, dtype=torch.float32)
            model.eval()
            for step in reversed(range(self.num_steps)):
                t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
                with torch.no_grad():
                    eps_theta = model(x, cond, t)
                beta_t = self.betas[step]
                alpha_t = self.alphas[step]
                alpha_bar_t = self.alpha_bars[step]
                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
                z = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
                x = coef1 * (x - coef2 * eps_theta) + torch.sqrt(beta_t) * z
            return x

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmokeDenoiser(input_dim=1, condition_dim=len(condition_cols)).to(device)
    scheduler = SmokeScheduler(num_steps=20, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    c_train_tensor = torch.tensor(C_train, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    final_train_loss = None
    batch_size = min(32, len(x_train_tensor))
    epochs = 2
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(x_train_tensor), device=device)
        epoch_losses = []
        model.train()
        for start in range(0, len(x_train_tensor), batch_size):
            idx = permutation[start:start + batch_size]
            x_batch = x_train_tensor[idx]
            c_batch = c_train_tensor[idx]
            t = torch.randint(0, scheduler.num_steps, (len(idx),), device=device)
            noise = torch.randn_like(x_batch)
            noisy = scheduler.q_sample(x_batch, t, noise)
            predicted = model(noisy, c_batch, t)
            loss = F.mse_loss(predicted, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_losses))
        final_train_loss = epoch_loss
        print(f"Smoke epoch {epoch}/{epochs} loss={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, paths["best_checkpoint"])

    if best_state is None:
        raise RuntimeError("Smoke training finished without a checkpoint.")
    torch.save(model.state_dict(), paths["last_checkpoint"])

    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    preview_count = min(8, len(C_test))
    preview_cond = torch.tensor(C_test[:preview_count], dtype=torch.float32, device=device)
    sample_scaled = scheduler.sample_reverse(model, preview_cond, shape=(preview_count, seq_len, 1)).detach().cpu().numpy()
    sample_unscaled = sample_scaled * target_scale.reshape(1, 1, -1) + target_mean.reshape(1, 1, -1)
    np.save(paths["sample_scaled"], sample_scaled)
    np.save(paths["sample_unscaled"], sample_unscaled)

    model_config = {
        "artifact_mode": "smoke",
        "input_dim": 1,
        "condition_dim": len(condition_cols),
        "seq_len": seq_len,
        "hidden_dim": 24,
        "time_dim": 24,
        "num_diffusion_steps": 20,
        "device": device,
        "seed": 42,
        "target_cols": target_cols,
        "condition_cols": condition_cols,
    }
    paths["model_config"].write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    summary = {
        "artifact_mode": "smoke",
        "num_epochs_completed": epochs,
        "final_train_loss": final_train_loss,
        "final_val_loss": None,
        "best_train_loss": best_loss,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "best_checkpoint": str(paths["best_checkpoint"]),
        "sample_unscaled": str(paths["sample_unscaled"]),
        "total_seconds": time.time() - start_time,
    }
    paths["training_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Smoke run complete.")
    print("Smoke scalers:", paths["scalers"])
    print("Smoke checkpoint:", paths["best_checkpoint"])
    print("Smoke generated sample:", paths["sample_unscaled"])
    print(f"Smoke runtime seconds: {summary['total_seconds']:.1f}")
    if enable_mlflow:
        tracking_dir = _log_smoke_mlflow_run(
            paths=paths,
            model_config=model_config,
            scaler_payload=scaler_payload,
            summary=summary,
        )
        print("MLflow tracking directory:", tracking_dir)
    else:
        print("MLflow logging disabled.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the upgraded counterfactual notebook pipeline in order.")
    parser.add_argument("--smoke", action="store_true", help="Run a small SPY-only smoke pipeline and write artifacts under counterfactual_data_build/smoke/.")
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable local MLflow logging for --smoke runs.")
    parser.add_argument("--start-at", help="Notebook filename to start at.", default=None)
    parser.add_argument("--stop-after", help="Notebook filename to stop after.", default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Per-cell timeout in seconds. Default is no timeout.")
    parser.add_argument("--kernel-name", default="python3", help="Jupyter kernel name to use.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.smoke:
        if args.start_at or args.stop_after:
            raise ValueError("--smoke cannot be combined with --start-at or --stop-after.")
        return _run_smoke_pipeline(enable_mlflow=not args.disable_mlflow)

    nbformat, NotebookClient = load_notebook_runtime()
    sequence = select_sequence(args.start_at, args.stop_after)

    print(f"Workspace: {ROOT}")
    print(f"Executed notebook output dir: {EXECUTED_DIR}")
    print("Notebook run order:")
    for name in sequence:
        print(f"- {name}")

    for name in sequence:
        notebook_path = ROOT / name
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        print()
        print(f"Running {name} ...")
        output_path = execute_notebook(nbformat, NotebookClient, notebook_path, args.timeout, args.kernel_name)
        print(f"Completed {name}")
        print(f"Saved executed notebook: {output_path}")

    print()
    print("Pipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
