from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("training_prep_for_upgraded_ddpm.ipynb")


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
        # Training Prep For The Upgraded DDPM

        This notebook converts the upgraded aligned dataset into train / validation / test tensors for the next retraining run.

        ## What it does

        - reads the aligned market + macro dataset
        - identifies target columns and dynamic conditioning columns
        - scales targets and conditioning features using train-only statistics
        - applies chronological train / val / test splits
        - builds rolling windows for the diffusion model
        - saves arrays, metadata tables, and scaler payloads

        The structure intentionally mirrors the earlier project notebook so we can compare the upgraded retrain against the original DDPM as cleanly as possible.
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
            "sklearn": "scikit-learn",
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
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from sklearn.preprocessing import StandardScaler

        plt.style.use("seaborn-v0_8-darkgrid")
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 220)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        """
    ),
    md(
        """
        ## Configuration

        This notebook expects the aligned dataset to have been produced already by:

        1. `release_aware_macro_loader.ipynb`
        2. `market_state_and_alignment_builder.ipynb`
        """
    ),
    code(
        """
        SPEC_PATH = Path("dataset_spec_v1.json")
        ARTIFACT_ROOT = Path.cwd() / "counterfactual_data_build"

        with open(SPEC_PATH, "r", encoding="utf-8") as handle:
            dataset_spec = json.load(handle)

        aligned_path = ARTIFACT_ROOT / Path(dataset_spec["loader_outputs"]["aligned_dataset_csv"])
        windows_dir = ARTIFACT_ROOT / "processed" / "windows"
        tables_dir = ARTIFACT_ROOT / "outputs" / "tables"

        windows_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        if not aligned_path.exists():
            raise FileNotFoundError(
                f"Missing aligned dataset: {aligned_path}. Run the market-state alignment notebook first."
            )

        seq_len = int(dataset_spec["training_layout"]["seq_len"])
        forecast_horizon = int(dataset_spec["training_layout"]["forecast_horizon"])
        split_ratios = dataset_spec["training_layout"]["split_ratios"]

        print("Aligned dataset:", aligned_path)
        print("Windows dir:", windows_dir)
        print("Tables dir:", tables_dir)
        print("Sequence length:", seq_len)
        print("Forecast horizon:", forecast_horizon)
        """
    ),
    md(
        """
        ## Load And Inspect The Aligned Dataset
        """
    ),
    code(
        """
        aligned_df = pd.read_csv(aligned_path)
        aligned_df["Date"] = pd.to_datetime(aligned_df["Date"]).dt.normalize()
        aligned_df = aligned_df.sort_values("Date").reset_index(drop=True)

        target_cols = [target["target_column"] for target in dataset_spec["targets"]]
        dynamic_condition_cols = [
            column
            for column in aligned_df.columns
            if column not in {"Date", *target_cols}
        ]

        required_cols = ["Date"] + target_cols + dynamic_condition_cols
        aligned_df = aligned_df[required_cols].dropna().reset_index(drop=True)

        print("Aligned shape:", aligned_df.shape)
        print("Target columns:", target_cols)
        print("Condition column count:", len(dynamic_condition_cols))
        display(aligned_df.tail())
        """
    ),
    md(
        """
        ## Chronological Split

        We keep the same chronological split logic as the original notebook:

        - train first
        - validation next
        - test last
        """
    ),
    code(
        """
        n_total = len(aligned_df)
        n_train = int(n_total * float(split_ratios["train"]))
        n_val = int(n_total * float(split_ratios["val"]))
        n_test = n_total - n_train - n_val

        train_df = aligned_df.iloc[:n_train].copy().reset_index(drop=True)
        val_df = aligned_df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
        test_df = aligned_df.iloc[n_train + n_val:].copy().reset_index(drop=True)

        split_summary = pd.DataFrame(
            [
                {
                    "split": "train",
                    "row_count": len(train_df),
                    "start_date": str(train_df["Date"].min().date()),
                    "end_date": str(train_df["Date"].max().date()),
                },
                {
                    "split": "val",
                    "row_count": len(val_df),
                    "start_date": str(val_df["Date"].min().date()),
                    "end_date": str(val_df["Date"].max().date()),
                },
                {
                    "split": "test",
                    "row_count": len(test_df),
                    "start_date": str(test_df["Date"].min().date()),
                    "end_date": str(test_df["Date"].max().date()),
                },
            ]
        )

        display(split_summary)
        """
    ),
    md(
        """
        ## Scale Targets And Dynamic Conditioning Features

        Train-only scaling keeps the future splits honest.
        """
    ),
    code(
        """
        target_scaler = StandardScaler()
        condition_scaler = StandardScaler()

        target_scaler.fit(train_df[target_cols])
        condition_scaler.fit(train_df[dynamic_condition_cols])

        def apply_scaling(frame):
            out = frame.copy()
            out[target_cols] = target_scaler.transform(out[target_cols])
            out[dynamic_condition_cols] = condition_scaler.transform(out[dynamic_condition_cols])
            return out

        train_scaled = apply_scaling(train_df)
        val_scaled = apply_scaling(val_df)
        test_scaled = apply_scaling(test_df)

        scaler_payload = {
            "target_cols": target_cols,
            "condition_cols": dynamic_condition_cols,
            "return_scaler_mean": target_scaler.mean_.tolist(),
            "return_scaler_scale": target_scaler.scale_.tolist(),
            "condition_scaler_mean": condition_scaler.mean_.tolist(),
            "condition_scaler_scale": condition_scaler.scale_.tolist(),
            "seq_len": seq_len,
            "forecast_horizon": forecast_horizon,
        }

        scaling_check_rows = []
        for split_name, frame in [
            ("train_scaled", train_scaled),
            ("val_scaled", val_scaled),
            ("test_scaled", test_scaled),
        ]:
            for column in target_cols[:3] + dynamic_condition_cols[:10]:
                scaling_check_rows.append(
                    {
                        "split": split_name,
                        "column": column,
                        "mean": float(frame[column].mean()),
                        "std": float(frame[column].std(ddof=0)),
                    }
                )

        scaling_check = pd.DataFrame(scaling_check_rows)
        display(scaling_check.head(20))
        """
    ),
    md(
        """
        ## Build Rolling Windows

        This mirrors the original pipeline structure:

        - `X`: scaled target history
        - `C`: scaled dynamic conditioning history
        - `meta`: end-date metadata for each sample
        """
    ),
    code(
        """
        def build_windows(split_df, split_name, seq_len, target_cols, condition_cols):
            x_windows = []
            c_windows = []
            end_dates = []

            target_values = split_df[target_cols].to_numpy(dtype=np.float32)
            condition_values = split_df[condition_cols].to_numpy(dtype=np.float32)
            dates = split_df["Date"].to_numpy()

            for end_idx in range(seq_len - 1, len(split_df)):
                start_idx = end_idx - seq_len + 1
                x_window = target_values[start_idx:end_idx + 1]
                c_window = condition_values[start_idx:end_idx + 1]

                if x_window.shape != (seq_len, len(target_cols)):
                    continue
                if c_window.shape != (seq_len, len(condition_cols)):
                    continue

                x_windows.append(x_window)
                c_windows.append(c_window)
                end_dates.append(pd.Timestamp(dates[end_idx]))

            if not x_windows:
                raise RuntimeError(f"No windows were created for split {split_name}")

            X = np.stack(x_windows).astype(np.float32)
            C = np.stack(c_windows).astype(np.float32)
            meta = pd.DataFrame(
                {
                    "split": split_name,
                    "window_end_date": end_dates,
                }
            )
            return X, C, meta


        X_train, C_train, meta_train = build_windows(train_scaled, "train", seq_len, target_cols, dynamic_condition_cols)
        X_val, C_val, meta_val = build_windows(val_scaled, "val", seq_len, target_cols, dynamic_condition_cols)
        X_test, C_test, meta_test = build_windows(test_scaled, "test", seq_len, target_cols, dynamic_condition_cols)

        window_shapes = pd.DataFrame(
            [
                {"array_name": "X_train", "shape": str(X_train.shape)},
                {"array_name": "C_train", "shape": str(C_train.shape)},
                {"array_name": "X_val", "shape": str(X_val.shape)},
                {"array_name": "C_val", "shape": str(C_val.shape)},
                {"array_name": "X_test", "shape": str(X_test.shape)},
                {"array_name": "C_test", "shape": str(C_test.shape)},
            ]
        )

        display(window_shapes)
        """
    ),
    md(
        """
        ## Save Arrays, Metadata, And Scalers
        """
    ),
    code(
        """
        np.save(windows_dir / "X_train.npy", X_train)
        np.save(windows_dir / "C_train.npy", C_train)
        np.save(windows_dir / "X_val.npy", X_val)
        np.save(windows_dir / "C_val.npy", C_val)
        np.save(windows_dir / "X_test.npy", X_test)
        np.save(windows_dir / "C_test.npy", C_test)

        meta_train.to_csv(windows_dir / "meta_train.csv", index=False)
        meta_val.to_csv(windows_dir / "meta_val.csv", index=False)
        meta_test.to_csv(windows_dir / "meta_test.csv", index=False)

        with open(windows_dir / "scalers.json", "w", encoding="utf-8") as handle:
            json.dump(scaler_payload, handle, indent=2)

        split_summary.to_csv(tables_dir / "time_split_summary.csv", index=False)
        window_shapes.to_csv(tables_dir / "window_array_shapes.csv", index=False)
        scaling_check.to_csv(tables_dir / "scaling_check_summary.csv", index=False)

        print("Saved arrays and metadata to:", windows_dir)
        print("Saved summaries to:", tables_dir)
        """
    ),
    md(
        """
        ## Diagnostics

        These plots help verify that the upgraded target set is stable enough before retraining.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        train_df.plot(x="Date", y=target_cols, ax=axes[0], linewidth=1.2)
        axes[0].set_title("Unscaled target return history")
        axes[0].set_ylabel("Log return")

        rolling_target_vol = train_df[target_cols].rolling(20).std(ddof=0) * np.sqrt(252)
        rolling_target_vol["Date"] = train_df["Date"]
        rolling_target_vol.plot(x="Date", y=target_cols, ax=axes[1], linewidth=1.2)
        axes[1].set_title("20-day target realized volatility")
        axes[1].set_ylabel("Annualized vol")

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
        print("Training prep complete.")
        print()
        print("Target columns:")
        for col in target_cols:
            print("-", col)
        print()
        print("Condition column count:", len(dynamic_condition_cols))
        print("First 15 condition columns:")
        for col in dynamic_condition_cols[:15]:
            print("-", col)
        print()
        print("Window shapes:")
        print(window_shapes.to_string(index=False))
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
