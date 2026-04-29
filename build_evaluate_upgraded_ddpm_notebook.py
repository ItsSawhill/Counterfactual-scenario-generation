from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("evaluate_upgraded_ddpm.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Evaluate The Upgraded DDPM

        This notebook evaluates the retrained upgraded DDPM against:

        - real test windows
        - a simple multivariate Gaussian baseline

        ## Outputs

        - generated DDPM and Gaussian baseline test windows
        - distribution diagnostics
        - time-series diagnostics
        - cross-asset correlation diagnostics
        - summary tables and comparison plots
        """
    ),
    code(
        """
        import importlib.util
        import subprocess
        import sys

        required_packages = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "scipy": "scipy",
            "torch": "torch",
            "IPython": "IPython",
        }

        missing_packages = [
            package_name
            for module_name, package_name in required_packages.items()
            if importlib.util.find_spec(module_name) is None
        ]

        if missing_packages:
            print("Installing missing packages:", ", ".join(missing_packages))
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        else:
            print("All required packages are already available.")
        """
    ),
    code(
        """
        import json
        import math
        import random
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from IPython.display import display
        from scipy.stats import ks_2samp, kurtosis, skew, wasserstein_distance

        plt.style.use("seaborn-v0_8-darkgrid")
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 220)
        pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
        """
    ),
    md(
        """
        ## Configuration
        """
    ),
    code(
        """
        ARTIFACT_ROOT = Path.cwd() / "counterfactual_data_build"
        WINDOWS_DIR = ARTIFACT_ROOT / "processed" / "windows"
        CHECKPOINT_DIR = ARTIFACT_ROOT / "outputs" / "checkpoints"
        TABLES_DIR = ARTIFACT_ROOT / "outputs" / "tables"
        SAMPLES_DIR = ARTIFACT_ROOT / "outputs" / "generated_samples"
        FIGURES_DIR = ARTIFACT_ROOT / "outputs" / "figures"

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        RUN_CONFIG = {
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "sample_batch_size": 64,
            "max_eval_windows": None,
        }

        random.seed(RUN_CONFIG["seed"])
        np.random.seed(RUN_CONFIG["seed"])
        torch.manual_seed(RUN_CONFIG["seed"])

        required_paths = {
            "X_train": WINDOWS_DIR / "X_train.npy",
            "X_test": WINDOWS_DIR / "X_test.npy",
            "C_test": WINDOWS_DIR / "C_test.npy",
            "scalers": WINDOWS_DIR / "scalers.json",
            "model_config": TABLES_DIR / "ddpm_upgraded_model_config.json",
            "checkpoint": CHECKPOINT_DIR / "conditional_ddpm_upgraded_best.pt",
        }

        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required artifacts:\\n" +
                "\\n".join(f"- {name}: {required_paths[name]}" for name in missing)
            )
        """
    ),
    code(
        """
        X_train = np.load(required_paths["X_train"]).astype(np.float32)
        X_test = np.load(required_paths["X_test"]).astype(np.float32)
        C_test = np.load(required_paths["C_test"]).astype(np.float32)

        with open(required_paths["scalers"], "r", encoding="utf-8") as handle:
            scalers = json.load(handle)
        with open(required_paths["model_config"], "r", encoding="utf-8") as handle:
            model_config = json.load(handle)

        target_cols = scalers["target_cols"]
        condition_cols = scalers["condition_cols"]
        return_mean = np.array(scalers["return_scaler_mean"], dtype=np.float64)
        return_scale = np.array(scalers["return_scaler_scale"], dtype=np.float64)

        if RUN_CONFIG["max_eval_windows"] is not None:
            eval_count = min(int(RUN_CONFIG["max_eval_windows"]), len(X_test))
            X_test = X_test[:eval_count]
            C_test = C_test[:eval_count]

        print("Train windows:", X_train.shape)
        print("Eval windows:", X_test.shape)
        print("Targets:", target_cols)
        """
    ),
    md(
        """
        ## Model Definition
        """
    ),
    code(
        """
        def sinusoidal_timestep_embedding(timesteps, dim):
            half_dim = dim // 2
            exponent = -math.log(10000.0) / max(half_dim - 1, 1)
            freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
            args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            if dim % 2 == 1:
                emb = F.pad(emb, (0, 1))
            return emb


        class ResidualTemporalBlock(nn.Module):
            def __init__(self, channels, time_dim, dilation, groups=8, dropout=0.05):
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
                self.norm1 = nn.GroupNorm(groups, channels)
                self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
                self.norm2 = nn.GroupNorm(groups, channels)
                self.time_proj = nn.Linear(time_dim, channels)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, t_emb):
                h = self.conv1(x)
                h = h + self.time_proj(t_emb).unsqueeze(-1)
                h = self.norm1(h)
                h = F.silu(h)
                h = self.dropout(h)
                h = self.conv2(h)
                h = self.norm2(h)
                h = F.silu(h)
                h = self.dropout(h)
                return x + h


        class ConditionalDiffusionModel(nn.Module):
            def __init__(
                self,
                input_dim,
                condition_dim,
                seq_len,
                hidden_dim=96,
                time_dim=192,
                num_blocks=8,
                dilations=(1, 2, 4, 8, 1, 2, 4, 8),
                dropout=0.05,
            ):
                super().__init__()
                self.time_dim = time_dim
                self.input_proj = nn.Conv1d(input_dim + condition_dim, hidden_dim, kernel_size=3, padding=1)
                self.cond_global_mlp = nn.Sequential(
                    nn.Linear(condition_dim, time_dim),
                    nn.SiLU(),
                    nn.Linear(time_dim, time_dim),
                )
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim, time_dim),
                    nn.SiLU(),
                    nn.Linear(time_dim, time_dim),
                )
                self.blocks = nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            channels=hidden_dim,
                            time_dim=time_dim,
                            dilation=dilations[i % len(dilations)],
                            groups=8,
                            dropout=dropout,
                        )
                        for i in range(num_blocks)
                    ]
                )
                self.output_head = nn.Sequential(
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
                )

            def forward(self, x, c, t):
                x = x.transpose(1, 2)
                c = c.transpose(1, 2)
                t_emb = sinusoidal_timestep_embedding(t, self.time_dim)
                c_global = c.mean(dim=2)
                t_emb = self.time_mlp(t_emb + self.cond_global_mlp(c_global))
                h = torch.cat([x, c], dim=1)
                h = self.input_proj(h)
                for block in self.blocks:
                    h = block(h, t_emb)
                out = self.output_head(h)
                return out.transpose(1, 2)


        class DiffusionScheduler:
            def __init__(self, num_steps=300, beta_start=1e-4, beta_end=1.5e-2, device="cpu"):
                self.num_steps = num_steps
                self.device = device
                self.betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
                self.alphas = 1.0 - self.betas
                self.alpha_bars = torch.cumprod(self.alphas, dim=0)

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
        """
    ),
    code(
        """
        model = ConditionalDiffusionModel(
            input_dim=int(model_config["input_dim"]),
            condition_dim=int(model_config["condition_dim"]),
            seq_len=int(model_config["seq_len"]),
            hidden_dim=int(model_config["hidden_dim"]),
            time_dim=int(model_config["time_dim"]),
            num_blocks=int(model_config["num_blocks"]),
            dilations=tuple(model_config["dilations"]),
            dropout=float(model_config["dropout"]),
        ).to(RUN_CONFIG["device"])

        state_dict = torch.load(required_paths["checkpoint"], map_location=RUN_CONFIG["device"])
        model.load_state_dict(state_dict)
        model.eval()

        scheduler = DiffusionScheduler(
            num_steps=int(model_config["num_diffusion_steps"]),
            beta_start=float(model_config["beta_start"]),
            beta_end=float(model_config["beta_end"]),
            device=RUN_CONFIG["device"],
        )

        print("Loaded upgraded DDPM checkpoint.")
        """
    ),
    md(
        """
        ## Generate DDPM And Gaussian Baseline Windows
        """
    ),
    code(
        """
        def generate_ddpm_windows(cond_np, batch_size=64):
            cond_tensor = torch.tensor(cond_np, dtype=torch.float32, device=RUN_CONFIG["device"])
            batches = []
            for start_idx in range(0, len(cond_tensor), batch_size):
                end_idx = min(start_idx + batch_size, len(cond_tensor))
                cond_batch = cond_tensor[start_idx:end_idx]
                shape = (
                    cond_batch.shape[0],
                    int(model_config["seq_len"]),
                    int(model_config["input_dim"]),
                )
                sample_batch = scheduler.sample_reverse(model, cond_batch, shape)
                batches.append(sample_batch.detach().cpu().numpy())
            return np.concatenate(batches, axis=0)


        def unscale_windows(scaled_windows):
            return scaled_windows * return_scale.reshape(1, 1, -1) + return_mean.reshape(1, 1, -1)


        ddpm_test_scaled = generate_ddpm_windows(C_test, batch_size=RUN_CONFIG["sample_batch_size"])
        ddpm_test_unscaled = unscale_windows(ddpm_test_scaled)

        real_test_unscaled = unscale_windows(X_test.astype(np.float64))
        train_unscaled = unscale_windows(X_train.astype(np.float64))

        flat_train = train_unscaled.reshape(-1, train_unscaled.shape[-1])
        gaussian_mean = flat_train.mean(axis=0)
        gaussian_cov = np.cov(flat_train, rowvar=False)
        gaussian_cov += np.eye(gaussian_cov.shape[0]) * 1e-8

        gaussian_test_unscaled = np.zeros_like(real_test_unscaled)
        for i in range(real_test_unscaled.shape[0]):
            gaussian_test_unscaled[i] = np.random.multivariate_normal(
                mean=gaussian_mean,
                cov=gaussian_cov,
                size=real_test_unscaled.shape[1],
            )

        np.save(SAMPLES_DIR / "real_upgraded_test_windows_unscaled.npy", real_test_unscaled)
        np.save(SAMPLES_DIR / "ddpm_upgraded_test_windows_unscaled.npy", ddpm_test_unscaled)
        np.save(SAMPLES_DIR / "gaussian_upgraded_test_windows_unscaled.npy", gaussian_test_unscaled)

        print("Saved evaluation sample windows to:", SAMPLES_DIR)
        """
    ),
    md(
        """
        ## Diagnostics
        """
    ),
    code(
        """
        def flatten_asset(windows, asset_idx):
            return windows[:, :, asset_idx].reshape(-1).astype(np.float64)


        def lag1_autocorr(series):
            series = np.asarray(series, dtype=np.float64)
            if len(series) < 3:
                return np.nan
            left = series[:-1]
            right = series[1:]
            if np.std(left) < 1e-12 or np.std(right) < 1e-12:
                return np.nan
            return float(np.corrcoef(left, right)[0, 1])


        def avg_window_lag1(windows, asset_idx, squared=False):
            values = []
            for i in range(windows.shape[0]):
                x = windows[i, :, asset_idx]
                if squared:
                    x = x ** 2
                ac = lag1_autocorr(x)
                if np.isfinite(ac):
                    values.append(ac)
            return float(np.mean(values)) if values else np.nan


        def distribution_metrics(real_x, synth_x):
            ks = ks_2samp(real_x, synth_x, alternative="two-sided", mode="auto")
            return {
                "mean_abs_error": float(abs(np.mean(real_x) - np.mean(synth_x))),
                "std_abs_error": float(abs(np.std(real_x, ddof=0) - np.std(synth_x, ddof=0))),
                "skew_abs_error": float(abs(skew(real_x, bias=False) - skew(synth_x, bias=False))),
                "kurtosis_abs_error": float(abs(kurtosis(real_x, fisher=True, bias=False) - kurtosis(synth_x, fisher=True, bias=False))),
                "q01_abs_error": float(abs(np.quantile(real_x, 0.01) - np.quantile(synth_x, 0.01))),
                "q05_abs_error": float(abs(np.quantile(real_x, 0.05) - np.quantile(synth_x, 0.05))),
                "q95_abs_error": float(abs(np.quantile(real_x, 0.95) - np.quantile(synth_x, 0.95))),
                "q99_abs_error": float(abs(np.quantile(real_x, 0.99) - np.quantile(synth_x, 0.99))),
                "wasserstein_distance": float(wasserstein_distance(real_x, synth_x)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }


        def corr_distance(real_windows, synth_windows):
            real_corr = pd.DataFrame(real_windows.reshape(-1, real_windows.shape[-1]), columns=target_cols).corr().values
            synth_corr = pd.DataFrame(synth_windows.reshape(-1, synth_windows.shape[-1]), columns=target_cols).corr().values
            upper_idx = np.triu_indices_from(real_corr, k=1)
            return float(np.mean(np.abs(real_corr[upper_idx] - synth_corr[upper_idx])))


        rows = []
        for asset_idx, asset in enumerate(target_cols):
            real_x = flatten_asset(real_test_unscaled, asset_idx)
            ddpm_x = flatten_asset(ddpm_test_unscaled, asset_idx)
            gaussian_x = flatten_asset(gaussian_test_unscaled, asset_idx)

            ddpm_metrics = distribution_metrics(real_x, ddpm_x)
            gaussian_metrics = distribution_metrics(real_x, gaussian_x)

            rows.append(
                {
                    "asset": asset,
                    "ddpm_mean_abs_error": ddpm_metrics["mean_abs_error"],
                    "gaussian_mean_abs_error": gaussian_metrics["mean_abs_error"],
                    "ddpm_std_abs_error": ddpm_metrics["std_abs_error"],
                    "gaussian_std_abs_error": gaussian_metrics["std_abs_error"],
                    "ddpm_skew_abs_error": ddpm_metrics["skew_abs_error"],
                    "gaussian_skew_abs_error": gaussian_metrics["skew_abs_error"],
                    "ddpm_kurtosis_abs_error": ddpm_metrics["kurtosis_abs_error"],
                    "gaussian_kurtosis_abs_error": gaussian_metrics["kurtosis_abs_error"],
                    "ddpm_q01_abs_error": ddpm_metrics["q01_abs_error"],
                    "gaussian_q01_abs_error": gaussian_metrics["q01_abs_error"],
                    "ddpm_q05_abs_error": ddpm_metrics["q05_abs_error"],
                    "gaussian_q05_abs_error": gaussian_metrics["q05_abs_error"],
                    "ddpm_q95_abs_error": ddpm_metrics["q95_abs_error"],
                    "gaussian_q95_abs_error": gaussian_metrics["q95_abs_error"],
                    "ddpm_q99_abs_error": ddpm_metrics["q99_abs_error"],
                    "gaussian_q99_abs_error": gaussian_metrics["q99_abs_error"],
                    "ddpm_wasserstein_distance": ddpm_metrics["wasserstein_distance"],
                    "gaussian_wasserstein_distance": gaussian_metrics["wasserstein_distance"],
                    "ddpm_avg_lag1": avg_window_lag1(ddpm_test_unscaled, asset_idx, squared=False),
                    "gaussian_avg_lag1": avg_window_lag1(gaussian_test_unscaled, asset_idx, squared=False),
                    "real_avg_lag1": avg_window_lag1(real_test_unscaled, asset_idx, squared=False),
                    "ddpm_avg_sq_lag1": avg_window_lag1(ddpm_test_unscaled, asset_idx, squared=True),
                    "gaussian_avg_sq_lag1": avg_window_lag1(gaussian_test_unscaled, asset_idx, squared=True),
                    "real_avg_sq_lag1": avg_window_lag1(real_test_unscaled, asset_idx, squared=True),
                }
            )

        summary_df = pd.DataFrame(rows)
        corr_df = pd.DataFrame(
            [
                {
                    "model": "ddpm_upgraded",
                    "mean_abs_corr_distance": corr_distance(real_test_unscaled, ddpm_test_unscaled),
                },
                {
                    "model": "gaussian_baseline",
                    "mean_abs_corr_distance": corr_distance(real_test_unscaled, gaussian_test_unscaled),
                },
            ]
        )

        summary_path = TABLES_DIR / "upgraded_ddpm_vs_gaussian_summary.csv"
        corr_path = TABLES_DIR / "upgraded_ddpm_corr_distance.csv"
        summary_df.to_csv(summary_path, index=False)
        corr_df.to_csv(corr_path, index=False)

        display(summary_df)
        display(corr_df)
        """
    ),
    md(
        """
        ## Plots
        """
    ),
    code(
        """
        for asset_idx, asset in enumerate(target_cols):
            real_x = flatten_asset(real_test_unscaled, asset_idx)
            ddpm_x = flatten_asset(ddpm_test_unscaled, asset_idx)
            gaussian_x = flatten_asset(gaussian_test_unscaled, asset_idx)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            axes[0].hist(real_x, bins=80, alpha=0.45, density=True, label="Real")
            axes[0].hist(ddpm_x, bins=80, alpha=0.45, density=True, label="DDPM")
            axes[0].hist(gaussian_x, bins=80, alpha=0.35, density=True, label="Gaussian")
            axes[0].set_title(f"{asset} return distribution")
            axes[0].legend(loc="best")

            sample_steps = np.arange(real_test_unscaled.shape[1])
            axes[1].plot(sample_steps, np.median(real_test_unscaled[:, :, asset_idx], axis=0), label="Real median", linewidth=2.2)
            axes[1].plot(sample_steps, np.median(ddpm_test_unscaled[:, :, asset_idx], axis=0), label="DDPM median", linewidth=2.2)
            axes[1].plot(sample_steps, np.median(gaussian_test_unscaled[:, :, asset_idx], axis=0), label="Gaussian median", linewidth=2.2)
            axes[1].set_title(f"{asset} median generated window")
            axes[1].legend(loc="best")

            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"{asset.lower()}_upgraded_eval.png", dpi=180, bbox_inches="tight")
            plt.show()
        """
    ),
    md(
        """
        ## Output Summary
        """
    ),
    code(
        """
        print("Evaluation complete.")
        print("Summary table:", summary_path)
        print("Correlation distance table:", corr_path)
        print("Figures directory:", FIGURES_DIR)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH.resolve()}")
