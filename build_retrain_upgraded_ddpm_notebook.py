from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("retrain_upgraded_ddpm.ipynb")


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
        # Retrain The Upgraded Conditional DDPM

        This notebook retrains the diffusion model on the upgraded cross-asset / release-aware dataset while staying as close as possible to the original project structure.

        ## Inputs

        - `X_train.npy`, `C_train.npy`
        - `X_val.npy`, `C_val.npy`
        - `X_test.npy`, `C_test.npy`
        - `scalers.json`

        ## Outputs

        - best and last checkpoints
        - model config JSON
        - training history CSV
        - training summary JSON
        - first generated sample windows from the upgraded test split

        The point of this notebook is to create the upgraded DDPM baseline before we layer on explicit shock encoders, retrieval, or event/news context.
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
        import copy
        import json
        import math
        import random
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from IPython.display import display
        from torch.utils.data import DataLoader, Dataset

        plt.style.use("seaborn-v0_8-darkgrid")
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 220)
        pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
        """
    ),
    md(
        """
        ## Configuration

        These defaults are chosen to stay close to the stronger version of your original DDPM while adapting to the larger upgraded target universe.
        """
    ),
    code(
        """
        SPEC_PATH = Path("dataset_spec_v1.json")
        ARTIFACT_ROOT = Path.cwd() / "counterfactual_data_build"

        WINDOWS_DIR = ARTIFACT_ROOT / "processed" / "windows"
        CHECKPOINT_DIR = ARTIFACT_ROOT / "outputs" / "checkpoints"
        TABLES_DIR = ARTIFACT_ROOT / "outputs" / "tables"
        SAMPLES_DIR = ARTIFACT_ROOT / "outputs" / "generated_samples"

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        RUN_CONFIG = {
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 64,
            "max_epochs": 35,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "early_stopping_patience": 8,
            "sample_preview_windows": 96,
        }

        random.seed(RUN_CONFIG["seed"])
        np.random.seed(RUN_CONFIG["seed"])
        torch.manual_seed(RUN_CONFIG["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RUN_CONFIG["seed"])

        print("Device:", RUN_CONFIG["device"])
        print("Windows dir:", WINDOWS_DIR)
        """
    ),
    code(
        """
        required_paths = {
            "X_train": WINDOWS_DIR / "X_train.npy",
            "C_train": WINDOWS_DIR / "C_train.npy",
            "X_val": WINDOWS_DIR / "X_val.npy",
            "C_val": WINDOWS_DIR / "C_val.npy",
            "X_test": WINDOWS_DIR / "X_test.npy",
            "C_test": WINDOWS_DIR / "C_test.npy",
            "scalers": WINDOWS_DIR / "scalers.json",
        }

        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required training-prep artifacts:\\n" +
                "\\n".join(f"- {name}: {required_paths[name]}" for name in missing)
            )

        X_train = np.load(required_paths["X_train"]).astype(np.float32)
        C_train = np.load(required_paths["C_train"]).astype(np.float32)
        X_val = np.load(required_paths["X_val"]).astype(np.float32)
        C_val = np.load(required_paths["C_val"]).astype(np.float32)
        X_test = np.load(required_paths["X_test"]).astype(np.float32)
        C_test = np.load(required_paths["C_test"]).astype(np.float32)

        with open(required_paths["scalers"], "r", encoding="utf-8") as handle:
            scalers = json.load(handle)

        target_cols = scalers["target_cols"]
        condition_cols = scalers["condition_cols"]
        return_mean = np.array(scalers["return_scaler_mean"], dtype=np.float64)
        return_scale = np.array(scalers["return_scaler_scale"], dtype=np.float64)

        print("X_train shape:", X_train.shape)
        print("C_train shape:", C_train.shape)
        print("Target count:", len(target_cols))
        print("Condition count:", len(condition_cols))
        """
    ),
    md(
        """
        ## Dataset Wrappers
        """
    ),
    code(
        """
        class WindowDataset(Dataset):
            def __init__(self, X, C):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.C = torch.tensor(C, dtype=torch.float32)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.C[idx]


        train_dataset = WindowDataset(X_train, C_train)
        val_dataset = WindowDataset(X_val, C_val)
        test_dataset = WindowDataset(X_test, C_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=RUN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(RUN_CONFIG["device"] == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=RUN_CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=(RUN_CONFIG["device"] == "cuda"),
        )

        print("Train batches:", len(train_loader))
        print("Val batches:", len(val_loader))
        """
    ),
    md(
        """
        ## Model Definition

        This keeps the same family as the stronger version of the original project:

        - temporal residual blocks
        - sinusoidal timestep embeddings
        - conditional fusion through the denoising backbone
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
                self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
                self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

            def q_sample(self, x0, t, noise):
                sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
                sqrt_omb = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
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
        """
    ),
    md(
        """
        ## Model Configuration
        """
    ),
    code(
        """
        model_config = {
            "input_dim": int(X_train.shape[2]),
            "condition_dim": int(C_train.shape[2]),
            "seq_len": int(X_train.shape[1]),
            "hidden_dim": 96,
            "time_dim": 192,
            "num_blocks": 8,
            "dilations": [1, 2, 4, 8, 1, 2, 4, 8],
            "dropout": 0.05,
            "num_diffusion_steps": 300,
            "beta_start": 1e-4,
            "beta_end": 1.5e-2,
            "device": RUN_CONFIG["device"],
            "seed": RUN_CONFIG["seed"],
            "target_cols": target_cols,
            "condition_cols": condition_cols,
        }

        model = ConditionalDiffusionModel(
            input_dim=model_config["input_dim"],
            condition_dim=model_config["condition_dim"],
            seq_len=model_config["seq_len"],
            hidden_dim=model_config["hidden_dim"],
            time_dim=model_config["time_dim"],
            num_blocks=model_config["num_blocks"],
            dilations=tuple(model_config["dilations"]),
            dropout=model_config["dropout"],
        ).to(RUN_CONFIG["device"])

        scheduler = DiffusionScheduler(
            num_steps=model_config["num_diffusion_steps"],
            beta_start=model_config["beta_start"],
            beta_end=model_config["beta_end"],
            device=RUN_CONFIG["device"],
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=RUN_CONFIG["learning_rate"],
            weight_decay=RUN_CONFIG["weight_decay"],
        )

        model_config_path = TABLES_DIR / "ddpm_upgraded_model_config.json"
        with open(model_config_path, "w", encoding="utf-8") as handle:
            json.dump(model_config, handle, indent=2)

        print("Model parameters:", sum(param.numel() for param in model.parameters()))
        print("Model config saved to:", model_config_path)
        """
    ),
    md(
        """
        ## Training Utilities
        """
    ),
    code(
        """
        def diffusion_loss(model, scheduler, x0, cond):
            batch_size = x0.shape[0]
            t = torch.randint(
                low=0,
                high=scheduler.num_steps,
                size=(batch_size,),
                device=x0.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x0)
            noisy_x = scheduler.q_sample(x0, t, noise)
            predicted_noise = model(noisy_x, cond, t)
            return F.mse_loss(predicted_noise, noise)


        def run_epoch(model, scheduler, dataloader, optimizer=None):
            is_train = optimizer is not None
            model.train(is_train)
            total_loss = 0.0
            total_count = 0

            for x_batch, c_batch in dataloader:
                x_batch = x_batch.to(RUN_CONFIG["device"], non_blocking=True)
                c_batch = c_batch.to(RUN_CONFIG["device"], non_blocking=True)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                loss = diffusion_loss(model, scheduler, x_batch, c_batch)

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), RUN_CONFIG["grad_clip_norm"])
                    optimizer.step()

                batch_size = x_batch.shape[0]
                total_loss += float(loss.detach().cpu()) * batch_size
                total_count += batch_size

            return total_loss / max(total_count, 1)
        """
    ),
    md(
        """
        ## Train The Upgraded DDPM
        """
    ),
    code(
        """
        best_val_loss = float("inf")
        best_epoch = -1
        best_state_dict = None
        patience_counter = 0
        history_rows = []

        best_checkpoint_path = CHECKPOINT_DIR / "conditional_ddpm_upgraded_best.pt"
        last_checkpoint_path = CHECKPOINT_DIR / "conditional_ddpm_upgraded_last.pt"

        train_start_time = time.time()

        for epoch in range(1, RUN_CONFIG["max_epochs"] + 1):
            epoch_start_time = time.time()
            train_loss = run_epoch(model, scheduler, train_loader, optimizer=optimizer)
            val_loss = run_epoch(model, scheduler, val_loader, optimizer=None)
            epoch_seconds = time.time() - epoch_start_time

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_seconds": epoch_seconds,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
                torch.save(best_state_dict, best_checkpoint_path)
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | seconds={epoch_seconds:.1f}"
            )

            if patience_counter >= RUN_CONFIG["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        torch.save(model.state_dict(), last_checkpoint_path)

        total_train_seconds = time.time() - train_start_time
        history_df = pd.DataFrame(history_rows)
        history_path = TABLES_DIR / "ddpm_upgraded_training_history.csv"
        history_df.to_csv(history_path, index=False)

        if best_state_dict is None:
            raise RuntimeError("Training finished without a best checkpoint.")

        model.load_state_dict(best_state_dict)
        model.eval()

        training_summary = {
            "device": RUN_CONFIG["device"],
            "num_epochs_completed": int(len(history_df)),
            "max_epochs_allowed": int(RUN_CONFIG["max_epochs"]),
            "batch_size": int(RUN_CONFIG["batch_size"]),
            "train_samples": int(len(train_dataset)),
            "val_samples": int(len(val_dataset)),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "final_train_loss": float(history_df["train_loss"].iloc[-1]),
            "final_val_loss": float(history_df["val_loss"].iloc[-1]),
            "total_train_seconds": float(total_train_seconds),
            "best_checkpoint": str(best_checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "history_csv": str(history_path),
        }

        summary_path = TABLES_DIR / "ddpm_upgraded_training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(training_summary, handle, indent=2)

        display(history_df.tail())
        print("Saved best checkpoint:", best_checkpoint_path)
        print("Saved last checkpoint:", last_checkpoint_path)
        print("Saved training history:", history_path)
        print("Saved training summary:", summary_path)
        """
    ),
    md(
        """
        ## Training Curves
        """
    ),
    code(
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss", linewidth=2.0)
        ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", linewidth=2.0)
        ax.axvline(best_epoch, linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
        ax.set_title("Upgraded DDPM training history")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE noise-prediction loss")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Generate A First Test-Split Sample

        This does not replace the full evaluation notebook, but it gives us the first upgraded generated windows artifact immediately after training.
        """
    ),
    code(
        """
        preview_count = min(RUN_CONFIG["sample_preview_windows"], len(C_test))
        preview_cond = torch.tensor(C_test[:preview_count], dtype=torch.float32, device=RUN_CONFIG["device"])

        with torch.no_grad():
            preview_scaled = scheduler.sample_reverse(
                model,
                preview_cond,
                shape=(preview_count, model_config["seq_len"], model_config["input_dim"]),
            ).detach().cpu().numpy()

        preview_unscaled = preview_scaled * return_scale.reshape(1, 1, -1) + return_mean.reshape(1, 1, -1)

        preview_scaled_path = SAMPLES_DIR / "ddpm_upgraded_test_windows_scaled.npy"
        preview_unscaled_path = SAMPLES_DIR / "ddpm_upgraded_test_windows_unscaled.npy"

        np.save(preview_scaled_path, preview_scaled)
        np.save(preview_unscaled_path, preview_unscaled)

        print("Saved preview scaled samples:", preview_scaled_path)
        print("Saved preview unscaled samples:", preview_unscaled_path)
        print("Preview sample shape:", preview_scaled.shape)
        """
    ),
    md(
        """
        ## Quick Sample Diagnostic
        """
    ),
    code(
        """
        asset_preview = target_cols[0]
        asset_idx = target_cols.index(asset_preview)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(min(25, preview_unscaled.shape[0])):
            ax.plot(preview_unscaled[i, :, asset_idx], alpha=0.18, color="#0f766e")
        ax.plot(np.median(preview_unscaled[:, :, asset_idx], axis=0), linewidth=2.5, color="#14b8a6", label="Median path")
        ax.set_title(f"Preview generated windows for {asset_preview}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Unscaled log return")
        ax.legend(loc="best")
        plt.tight_layout()
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
        print("Retraining complete.")
        print()
        print("Best epoch:", best_epoch)
        print("Best validation loss:", f"{best_val_loss:.6f}")
        print("Model config:", model_config_path)
        print("Best checkpoint:", best_checkpoint_path)
        print("Last checkpoint:", last_checkpoint_path)
        print("Training history:", history_path)
        print("Training summary:", summary_path)
        print("Generated preview (scaled):", preview_scaled_path)
        print("Generated preview (unscaled):", preview_unscaled_path)
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
