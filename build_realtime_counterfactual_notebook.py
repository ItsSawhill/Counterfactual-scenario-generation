from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("realtime_counterfactual_generation_system.ipynb")


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
        # Real-Time Counterfactual Generation With A Conditional Diffusion Model

        This notebook adapts the trained conditional DDPM from your project into a live counterfactual system. It keeps the core diffusion-model inference logic from the finished project, replaces the old test-split conditioning input with the latest market and macro data, and then generates baseline and shocked scenarios from the current state.

        ## What this notebook does

        - pulls the latest ETF prices with `yfinance`
        - pulls the latest published macro series from FRED
        - rebuilds the most recent conditioning window in the same feature order used during training
        - loads the trained diffusion checkpoint, scaler payload, and model config
        - generates baseline and counterfactual return paths from the current live macro state
        - converts generated log returns into price paths and summarizes terminal risk

        ## Important note

        This notebook assumes you already trained the model and still have the saved artifacts:

        - `scalers.json`
        - `ddpm_improved_model_config.json` or `model_config.json`
        - `conditional_ddpm_improved_best.pt`

        Update the artifact path in the configuration cell before running.
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
            "yfinance": "yfinance",
            "pandas_datareader": "pandas_datareader",
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
        import warnings
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import yfinance as yf
        from IPython.display import display
        from pandas_datareader.data import DataReader
        from scipy.stats import kurtosis, skew

        warnings.filterwarnings("ignore")
        plt.style.use("seaborn-v0_8-darkgrid")
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.width", 180)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        """
    ),
    md(
        """
        ## Configuration

        Set the artifact root to the folder that contains the outputs from your trained project. The directory should contain the `data/processed/windows` and `outputs/checkpoints` folders from the original notebook pipeline.
        """
    ),
    code(
        """
        RUN_CONFIG = {
            "artifact_root": Path(r"C:\\path\\to\\financial_diffusion_final_project"),
            "market_tickers": ["SPY", "QQQ", "GLD"],
            "history_start": "2015-01-01",
            "generated_paths": 160,
            "forecast_horizon": 30,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "random_seed": 42,
            "output_dir": Path.cwd() / "live_counterfactual_outputs",
        }

        torch.manual_seed(RUN_CONFIG["random_seed"])
        np.random.seed(RUN_CONFIG["random_seed"])

        ARTIFACT_ROOT = RUN_CONFIG["artifact_root"]
        WINDOWS_DIR = ARTIFACT_ROOT / "data" / "processed" / "windows"
        TABLES_DIR = ARTIFACT_ROOT / "outputs" / "tables"
        CHECKPOINT_DIR = ARTIFACT_ROOT / "outputs" / "checkpoints"
        OUTPUT_DIR = RUN_CONFIG["output_dir"]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Artifact root:", ARTIFACT_ROOT)
        print("Output dir:", OUTPUT_DIR)
        print("Device:", RUN_CONFIG["device"])
        """
    ),
    code(
        """
        candidate_model_configs = [
            TABLES_DIR / "ddpm_improved_model_config.json",
            TABLES_DIR / "model_config.json",
        ]
        model_config_path = next((path for path in candidate_model_configs if path.exists()), None)

        candidate_checkpoints = [
            CHECKPOINT_DIR / "conditional_ddpm_improved_best.pt",
            CHECKPOINT_DIR / "conditional_ddpm_best.pt",
            CHECKPOINT_DIR / "conditional_ddpm_init.pt",
        ]
        checkpoint_path = next((path for path in candidate_checkpoints if path.exists()), None)

        scalers_path = WINDOWS_DIR / "scalers.json"

        required_paths = {
            "scalers": scalers_path,
            "model_config": model_config_path,
            "checkpoint": checkpoint_path,
        }

        missing = [name for name, path in required_paths.items() if path is None or not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required artifacts. Update RUN_CONFIG['artifact_root'] so these files exist:\\n"
                f"- scalers: {scalers_path}\\n"
                f"- model config candidates: {candidate_model_configs}\\n"
                f"- checkpoint candidates: {candidate_checkpoints}"
            )

        with open(scalers_path, "r", encoding="utf-8") as handle:
            scalers = json.load(handle)

        with open(model_config_path, "r", encoding="utf-8") as handle:
            model_config = json.load(handle)

        target_cols = scalers["target_cols"]
        condition_cols = scalers["condition_cols"]
        return_mean = np.array(scalers["return_scaler_mean"], dtype=np.float64)
        return_scale = np.array(scalers["return_scaler_scale"], dtype=np.float64)
        macro_mean = np.array(scalers["macro_scaler_mean"], dtype=np.float64)
        macro_scale = np.array(scalers["macro_scaler_scale"], dtype=np.float64)

        cond_index = {name: idx for idx, name in enumerate(condition_cols)}
        asset_names = [col.replace("_log_return", "") for col in target_cols]

        print("Loaded target columns:", target_cols)
        print("Loaded condition columns:", condition_cols)
        print("Model config path:", model_config_path)
        print("Checkpoint path:", checkpoint_path)
        """
    ),
    md(
        """
        ## Live Data Download

        The original project trained on daily ETF log returns plus aligned macro features. This section rebuilds that same feature table from the latest available market and macro observations.

        Notes:

        - `SPY`, `QQQ`, and `GLD` come from Yahoo Finance.
        - macro series come from FRED and are forward-filled to trading dates
        - monthly macro releases such as CPI and unemployment remain publication-lagged, which is expected in a real-time system
        """
    ),
    code(
        """
        def download_adjusted_close(tickers, start_date, end_date):
            raw = yf.download(
                tickers,
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if raw is None or len(raw) == 0:
                raise RuntimeError("No market data returned from yfinance.")

            if isinstance(raw.columns, pd.MultiIndex):
                if "Adj Close" in raw.columns.get_level_values(0):
                    prices = raw["Adj Close"].copy()
                elif "Close" in raw.columns.get_level_values(0):
                    prices = raw["Close"].copy()
                else:
                    raise RuntimeError(f"Unexpected yfinance columns: {raw.columns}")
            else:
                price_column = "Adj Close" if "Adj Close" in raw.columns else "Close"
                prices = raw[[price_column]].copy()
                prices.columns = tickers

            prices.index = pd.to_datetime(prices.index).normalize()
            prices = prices.sort_index()
            prices = prices.dropna(how="all")
            prices = prices.ffill().dropna()
            return prices


        def download_fred_series(series_id, start_date, end_date):
            try:
                frame = DataReader(series_id, "fred", start_date, end_date)
                frame = frame.reset_index().rename(columns={"DATE": "Date", series_id: "Value"})
            except Exception:
                csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                frame = pd.read_csv(csv_url)
                frame = frame.rename(columns={"DATE": "Date", series_id: "Value"})
                frame["Date"] = pd.to_datetime(frame["Date"])
                frame = frame[(frame["Date"] >= pd.Timestamp(start_date)) & (frame["Date"] <= pd.Timestamp(end_date))]

            frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
            frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
            frame = frame.dropna().sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            return frame.reset_index(drop=True)


        def reindex_macro_to_market_dates(macro_df, market_dates, output_name):
            series = macro_df.set_index("Date")["Value"].sort_index()
            series = series[~series.index.duplicated(keep="last")]
            series = series.reindex(series.index.union(market_dates)).sort_index().ffill()
            series = series.reindex(market_dates).ffill().bfill()
            return series.rename(output_name)


        end_date = pd.Timestamp.today().normalize()
        market_prices = download_adjusted_close(
            RUN_CONFIG["market_tickers"],
            RUN_CONFIG["history_start"],
            end_date.strftime("%Y-%m-%d"),
        )
        market_dates = pd.DatetimeIndex(market_prices.index)

        macro_frames = {}
        for series_id in condition_cols:
            macro_frames[series_id] = download_fred_series(
                series_id,
                RUN_CONFIG["history_start"],
                end_date.strftime("%Y-%m-%d"),
            )

        macro_aligned = [
            reindex_macro_to_market_dates(macro_frames[series_id], market_dates, series_id)
            for series_id in condition_cols
        ]

        returns_df = np.log(market_prices / market_prices.shift(1)).dropna()
        returns_df.columns = [f"{ticker}_log_return" for ticker in returns_df.columns]

        macro_df = pd.concat(macro_aligned, axis=1)
        aligned_live_df = returns_df.join(macro_df, how="left").dropna().reset_index()
        aligned_live_df = aligned_live_df.rename(columns={"index": "Date"}).sort_values("Date").reset_index(drop=True)

        live_snapshot = aligned_live_df.tail(5).copy()
        live_snapshot.to_csv(OUTPUT_DIR / "live_aligned_tail.csv", index=False)

        print("Aligned live frame shape:", aligned_live_df.shape)
        print("Latest aligned date:", aligned_live_df["Date"].max())
        display(live_snapshot)
        """
    ),
    code(
        """
        model_seq_len = int(model_config["seq_len"])
        model_horizon = int(model_config["seq_len"])
        forecast_horizon = min(int(RUN_CONFIG["forecast_horizon"]), model_horizon)

        if forecast_horizon < int(RUN_CONFIG["forecast_horizon"]):
            print(
                f"Requested horizon {RUN_CONFIG['forecast_horizon']} exceeds the trained model horizon "
                f"of {model_horizon}. Using {forecast_horizon}."
            )

        if len(aligned_live_df) < model_seq_len:
            raise RuntimeError(
                f"Need at least {model_seq_len} aligned observations, but only found {len(aligned_live_df)}."
            )

        latest_window_df = aligned_live_df.tail(model_seq_len).copy().reset_index(drop=True)
        latest_prices = market_prices.loc[latest_window_df["Date"].iloc[-1], asset_names].astype(float)

        raw_macro_window = latest_window_df[condition_cols].to_numpy(dtype=np.float64)
        standardized_macro_window = (raw_macro_window - macro_mean.reshape(1, -1)) / macro_scale.reshape(1, -1)

        live_context_summary = pd.DataFrame(
            {
                "series": condition_cols,
                "latest_value": raw_macro_window[-1],
                "train_mean": macro_mean,
                "train_scale": macro_scale,
                "latest_zscore": standardized_macro_window[-1],
            }
        )

        display(live_context_summary)
        print("Latest prices:")
        display(latest_prices.rename("spot_price").to_frame())
        """
    ),
    md(
        """
        ## Scenario Builder

        Each scenario starts from the latest observed macro window and then applies either:

        - an additive level shift such as `+1.0` percentage points to a rate series
        - a proportional shift such as `+1%` to CPI level

        This keeps the conditioning layout consistent with the trained model while letting you define real-time counterfactuals from the latest state.
        """
    ),
    code(
        """
        def apply_delta(raw_window, series_name, value):
            updated = raw_window.copy()
            updated[:, cond_index[series_name]] = updated[:, cond_index[series_name]] + float(value)
            return updated


        def apply_percent(raw_window, series_name, value):
            updated = raw_window.copy()
            updated[:, cond_index[series_name]] = updated[:, cond_index[series_name]] * (1.0 + float(value))
            return updated


        def apply_level(raw_window, series_name, value):
            updated = raw_window.copy()
            updated[:, cond_index[series_name]] = float(value)
            return updated


        def apply_shocks(raw_window, shocks):
            updated = raw_window.copy()
            for shock in shocks:
                mode = shock["mode"]
                series = shock["series"]
                value = shock["value"]
                if mode == "delta":
                    updated = apply_delta(updated, series, value)
                elif mode == "percent":
                    updated = apply_percent(updated, series, value)
                elif mode == "level":
                    updated = apply_level(updated, series, value)
                else:
                    raise ValueError(f"Unsupported shock mode: {mode}")
            return updated


        SCENARIOS = {
            "Baseline": [],
            "Hawkish Policy Shock": [
                {"series": "FEDFUNDS", "mode": "delta", "value": 1.00},
                {"series": "DGS10", "mode": "delta", "value": 0.50},
            ],
            "Sticky Inflation": [
                {"series": "CPIAUCSL", "mode": "percent", "value": 0.01},
                {"series": "VIXCLS", "mode": "delta", "value": 4.0},
            ],
            "Growth Scare": [
                {"series": "UNRATE", "mode": "delta", "value": 1.00},
                {"series": "VIXCLS", "mode": "delta", "value": 8.0},
                {"series": "DGS10", "mode": "delta", "value": -0.40},
            ],
            "Risk Compression": [
                {"series": "VIXCLS", "mode": "delta", "value": -5.0},
                {"series": "FEDFUNDS", "mode": "delta", "value": -0.50},
            ],
        }

        scenario_manifest_rows = []
        scenario_raw_windows = {}
        scenario_scaled_conditions = {}

        for scenario_name, shocks in SCENARIOS.items():
            scenario_window = apply_shocks(raw_macro_window, shocks)
            scenario_raw_windows[scenario_name] = scenario_window

            scaled_window = (scenario_window - macro_mean.reshape(1, -1)) / macro_scale.reshape(1, -1)
            scenario_scaled_conditions[scenario_name] = np.repeat(
                scaled_window[None, :, :],
                RUN_CONFIG["generated_paths"],
                axis=0,
            ).astype(np.float32)

            latest_snapshot = scenario_window[-1]
            row = {"scenario": scenario_name}
            for idx, col in enumerate(condition_cols):
                row[col] = latest_snapshot[idx]
            scenario_manifest_rows.append(row)

        scenario_manifest = pd.DataFrame(scenario_manifest_rows)
        display(scenario_manifest)
        """
    ),
    md(
        """
        ## Diffusion Model Definition

        This cell mirrors the improved architecture from the finished project notebook and loads the trained checkpoint for generation.
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
                input_dim=3,
                condition_dim=5,
                seq_len=30,
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


        inference_model = ConditionalDiffusionModel(
            input_dim=int(model_config["input_dim"]),
            condition_dim=int(model_config["condition_dim"]),
            seq_len=int(model_config["seq_len"]),
            hidden_dim=int(model_config["hidden_dim"]),
            time_dim=int(model_config["time_dim"]),
            num_blocks=int(model_config["num_blocks"]),
            dilations=tuple(model_config["dilations"]),
            dropout=float(model_config["dropout"]),
        ).to(RUN_CONFIG["device"])

        state_dict = torch.load(checkpoint_path, map_location=RUN_CONFIG["device"])
        inference_model.load_state_dict(state_dict)
        inference_model.eval()

        scheduler = DiffusionScheduler(
            num_steps=int(model_config["num_diffusion_steps"]),
            beta_start=float(model_config["beta_start"]),
            beta_end=float(model_config["beta_end"]),
            device=RUN_CONFIG["device"],
        )

        print("Loaded diffusion model successfully.")
        print("Parameter count:", sum(param.numel() for param in inference_model.parameters()))
        """
    ),
    md(
        """
        ## Generate Live Counterfactual Paths

        The model generates standardized return windows conditioned on the live macro state. Those returns are then unscaled and compounded from the latest observed prices to create scenario-specific price paths.
        """
    ),
    code(
        """
        def generate_return_windows(cond_np, batch_size=64):
            cond_tensor = torch.tensor(cond_np, dtype=torch.float32, device=RUN_CONFIG["device"])
            batches = []
            for start_idx in range(0, len(cond_tensor), batch_size):
                end_idx = min(start_idx + batch_size, len(cond_tensor))
                cond_batch = cond_tensor[start_idx:end_idx]
                sample_shape = (
                    cond_batch.shape[0],
                    int(model_config["seq_len"]),
                    int(model_config["input_dim"]),
                )
                sample_batch = scheduler.sample_reverse(inference_model, cond_batch, sample_shape)
                batches.append(sample_batch.detach().cpu().numpy())

            scaled = np.concatenate(batches, axis=0)
            unscaled = scaled * return_scale.reshape(1, 1, -1) + return_mean.reshape(1, 1, -1)
            return scaled, unscaled


        def returns_to_price_paths(return_windows, latest_price_series):
            price_paths = {}
            for asset_idx, asset in enumerate(asset_names):
                log_returns = return_windows[:, :forecast_horizon, asset_idx]
                cumulative_log_returns = np.cumsum(log_returns, axis=1)
                start_price = float(latest_price_series[asset])
                price_paths[asset] = start_price * np.exp(cumulative_log_returns)
            return price_paths


        generated_returns = {}
        generated_price_paths = {}

        for scenario_name, cond_np in scenario_scaled_conditions.items():
            _, unscaled_returns = generate_return_windows(cond_np)
            generated_returns[scenario_name] = unscaled_returns[:, :forecast_horizon, :]
            generated_price_paths[scenario_name] = returns_to_price_paths(
                generated_returns[scenario_name],
                latest_prices,
            )

        print("Generated scenarios:", list(generated_price_paths.keys()))
        """
    ),
    md(
        """
        ## Scenario Analytics

        These diagnostics are designed for counterfactual decision support rather than just visualization. They summarize terminal return, downside risk, tail loss, and distribution shape for each asset under each live scenario.
        """
    ),
    code(
        """
        def cvar(values, alpha=0.05):
            threshold = np.quantile(values, alpha)
            tail = values[values <= threshold]
            return float(tail.mean()) if len(tail) else float("nan")


        def summarize_scenario_paths(price_paths_dict, scenario_name):
            rows = []
            for asset in asset_names:
                start_price = float(latest_prices[asset])
                terminal_prices = price_paths_dict[asset][:, -1]
                terminal_returns = terminal_prices / start_price - 1.0

                rows.append(
                    {
                        "scenario": scenario_name,
                        "asset": asset,
                        "median_terminal_return": float(np.median(terminal_returns)),
                        "mean_terminal_return": float(np.mean(terminal_returns)),
                        "downside_probability": float(np.mean(terminal_returns < 0.0)),
                        "var_5": float(np.quantile(terminal_returns, 0.05)),
                        "var_1": float(np.quantile(terminal_returns, 0.01)),
                        "cvar_5": cvar(terminal_returns, alpha=0.05),
                        "std_terminal_return": float(np.std(terminal_returns, ddof=0)),
                        "skew_terminal_return": float(skew(terminal_returns, bias=False)),
                        "kurtosis_terminal_return": float(kurtosis(terminal_returns, fisher=True, bias=False)),
                        "q10_terminal_price": float(np.quantile(terminal_prices, 0.10)),
                        "median_terminal_price": float(np.quantile(terminal_prices, 0.50)),
                        "q90_terminal_price": float(np.quantile(terminal_prices, 0.90)),
                    }
                )
            return pd.DataFrame(rows)


        scenario_summary_frames = [
            summarize_scenario_paths(paths, scenario_name)
            for scenario_name, paths in generated_price_paths.items()
        ]
        scenario_summary = pd.concat(scenario_summary_frames, ignore_index=True)

        baseline_summary = (
            scenario_summary[scenario_summary["scenario"] == "Baseline"]
            .rename(
                columns={
                    "median_terminal_return": "baseline_median_terminal_return",
                    "mean_terminal_return": "baseline_mean_terminal_return",
                    "downside_probability": "baseline_downside_probability",
                }
            )[["asset", "baseline_median_terminal_return", "baseline_mean_terminal_return", "baseline_downside_probability"]]
        )

        scenario_comparison = scenario_summary.merge(baseline_summary, on="asset", how="left")
        scenario_comparison["delta_vs_baseline_mean_return"] = (
            scenario_comparison["mean_terminal_return"] - scenario_comparison["baseline_mean_terminal_return"]
        )
        scenario_comparison["delta_vs_baseline_downside"] = (
            scenario_comparison["downside_probability"] - scenario_comparison["baseline_downside_probability"]
        )

        scenario_summary.to_csv(OUTPUT_DIR / "live_scenario_summary.csv", index=False)
        scenario_comparison.to_csv(OUTPUT_DIR / "live_scenario_comparison.csv", index=False)

        display(scenario_summary)
        display(scenario_comparison[[
            "scenario",
            "asset",
            "mean_terminal_return",
            "downside_probability",
            "var_5",
            "cvar_5",
            "delta_vs_baseline_mean_return",
            "delta_vs_baseline_downside",
        ]])
        """
    ),
    md(
        """
        ## Visualization

        The charts below show the median forecast and a central confidence band for each scenario, starting from the latest observed price.
        """
    ),
    code(
        """
        def plot_asset_scenarios(asset_name, quantile_low=0.10, quantile_high=0.90):
            fig, ax = plt.subplots(figsize=(12, 6))
            horizon_index = np.arange(1, forecast_horizon + 1)

            for scenario_name, scenario_paths in generated_price_paths.items():
                asset_paths = scenario_paths[asset_name]
                median_path = np.quantile(asset_paths, 0.50, axis=0)
                lower_band = np.quantile(asset_paths, quantile_low, axis=0)
                upper_band = np.quantile(asset_paths, quantile_high, axis=0)

                ax.plot(horizon_index, median_path, linewidth=2.2, label=scenario_name)
                ax.fill_between(horizon_index, lower_band, upper_band, alpha=0.12)

            ax.axhline(float(latest_prices[asset_name]), linestyle="--", linewidth=1.2, color="black", alpha=0.55)
            ax.set_title(f"{asset_name} live counterfactual forecast")
            ax.set_xlabel("Forecast day")
            ax.set_ylabel("Price")
            ax.legend(loc="best")
            plt.tight_layout()
            return fig


        for asset in asset_names:
            figure = plot_asset_scenarios(asset)
            figure.savefig(OUTPUT_DIR / f"{asset.lower()}_live_counterfactual_paths.png", dpi=180, bbox_inches="tight")
            plt.show()
        """
    ),
    md(
        """
        ## Scenario Snapshot

        This final cell makes it easy to inspect the latest observed macro values beside the shocked endpoint for each live scenario.
        """
    ),
    code(
        """
        latest_observed = pd.DataFrame(
            {
                "series": condition_cols,
                "latest_observed": raw_macro_window[-1],
            }
        )

        scenario_endpoints = scenario_manifest.melt(
            id_vars=["scenario"],
            value_vars=condition_cols,
            var_name="series",
            value_name="scenario_value",
        )

        snapshot_table = scenario_endpoints.merge(latest_observed, on="series", how="left")
        snapshot_table["difference"] = snapshot_table["scenario_value"] - snapshot_table["latest_observed"]
        snapshot_table.to_csv(OUTPUT_DIR / "scenario_macro_snapshot.csv", index=False)

        display(snapshot_table)
        print("Saved outputs to:", OUTPUT_DIR)
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
