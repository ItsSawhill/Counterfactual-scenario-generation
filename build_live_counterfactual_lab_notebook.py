from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("live_counterfactual_lab_v1.ipynb")


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
        # Live Counterfactual Lab V1

        This notebook uses the upgraded trained DDPM to generate live counterfactual scenarios from the latest market and macro state.

        ## Inputs

        - upgraded checkpoint and model config
        - upgraded scaler payload
        - `scenario_definitions_v1.csv`
        - live market prices from Yahoo Finance
        - live macro series from FRED or ALFRED-backed fallback logic

        ## Outputs

        - scenario summary table
        - generated path plots for the upgraded target universe
        - saved live scenario outputs under `outputs/live_counterfactuals`
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
            "requests": "requests",
            "yfinance": "yfinance",
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
        import os
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import requests
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import yfinance as yf
        from IPython.display import display
        from pandas.tseries.offsets import MonthEnd

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
        SPEC_PATH = Path("dataset_spec_v1.json")
        SCENARIO_PATH = Path("scenario_definitions_v1.csv")
        ARTIFACT_ROOT = Path.cwd() / "counterfactual_data_build"

        WINDOWS_DIR = ARTIFACT_ROOT / "processed" / "windows"
        TABLES_DIR = ARTIFACT_ROOT / "outputs" / "tables"
        CHECKPOINT_DIR = ARTIFACT_ROOT / "outputs" / "checkpoints"
        LIVE_OUTPUT_DIR = ARTIFACT_ROOT / "outputs" / "live_counterfactuals"
        LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

        RUN_CONFIG = {
            "history_start": "2015-01-01",
            "generated_paths": 160,
            "forecast_horizon": 30,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        with open(SPEC_PATH, "r", encoding="utf-8") as handle:
            dataset_spec = json.load(handle)
        scenarios_df = pd.read_csv(SCENARIO_PATH)

        required_paths = {
            "scalers": WINDOWS_DIR / "scalers.json",
            "model_config": TABLES_DIR / "ddpm_upgraded_model_config.json",
            "checkpoint": CHECKPOINT_DIR / "conditional_ddpm_upgraded_best.pt",
        }
        missing = [name for name, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required live inference artifacts:\\n" +
                "\\n".join(f"- {name}: {required_paths[name]}" for name in missing)
            )
        """
    ),
    code(
        """
        with open(required_paths["scalers"], "r", encoding="utf-8") as handle:
            scalers = json.load(handle)
        with open(required_paths["model_config"], "r", encoding="utf-8") as handle:
            model_config = json.load(handle)

        target_specs = dataset_spec["targets"]
        macro_specs = dataset_spec["macro_series"]
        target_cols = scalers["target_cols"]
        condition_cols = scalers["condition_cols"]

        return_mean = np.array(scalers["return_scaler_mean"], dtype=np.float64)
        return_scale = np.array(scalers["return_scaler_scale"], dtype=np.float64)
        condition_mean = np.array(scalers["condition_scaler_mean"], dtype=np.float64)
        condition_scale = np.array(scalers["condition_scaler_scale"], dtype=np.float64)
        condition_mean_map = dict(zip(condition_cols, condition_mean))

        target_name_map = {
            spec["normalized_name"]: spec["target_column"]
            for spec in target_specs
        }
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
            def __init__(self, input_dim, condition_dim, seq_len, hidden_dim=96, time_dim=192, num_blocks=8, dilations=(1, 2, 4, 8, 1, 2, 4, 8), dropout=0.05):
                super().__init__()
                self.time_dim = time_dim
                self.input_proj = nn.Conv1d(input_dim + condition_dim, hidden_dim, kernel_size=3, padding=1)
                self.cond_global_mlp = nn.Sequential(nn.Linear(condition_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
                self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
                self.blocks = nn.ModuleList([
                    ResidualTemporalBlock(channels=hidden_dim, time_dim=time_dim, dilation=dilations[i % len(dilations)], groups=8, dropout=dropout)
                    for i in range(num_blocks)
                ])
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
        """
    ),
    md(
        """
        ## Build Live Market And Macro State
        """
    ),
    code(
        """
        def choose_price_frame(raw_download):
            if isinstance(raw_download.columns, pd.MultiIndex):
                level0 = list(raw_download.columns.get_level_values(0).unique())
                if "Adj Close" in level0:
                    prices = raw_download["Adj Close"].copy()
                else:
                    prices = raw_download["Close"].copy()
                volumes = raw_download["Volume"].copy() if "Volume" in level0 else None
            else:
                price_col = "Adj Close" if "Adj Close" in raw_download.columns else "Close"
                prices = raw_download[[price_col]].copy()
                prices.columns = ["asset"]
                volumes = raw_download[["Volume"]].copy() if "Volume" in raw_download.columns else None
            prices.index = pd.to_datetime(prices.index).normalize()
            prices = prices.sort_index().ffill()
            if volumes is not None:
                volumes.index = pd.to_datetime(volumes.index).normalize()
                volumes = volumes.sort_index()
            return prices, volumes


        def download_targets():
            tickers = [spec["symbol"] for spec in target_specs]
            raw = yf.download(
                tickers,
                start=RUN_CONFIG["history_start"],
                end=(pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            prices, volumes = choose_price_frame(raw)
            rename_map = {spec["symbol"]: spec["normalized_name"] for spec in target_specs}
            prices = prices.rename(columns=rename_map)
            if volumes is not None:
                volumes = volumes.rename(columns=rename_map)
            return prices, volumes


        def fetch_fred_public_history(series_id):
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            frame = pd.read_csv(url)
            frame = frame.rename(columns={"DATE": "observation_date", series_id: "value"})
            frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.normalize()
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["value"]).sort_values("observation_date").reset_index(drop=True)
            return frame


        def infer_release_date(observation_date, spec):
            ts = pd.Timestamp(observation_date).normalize()
            if spec["observation_anchor"] == "same_day":
                base = ts
            else:
                base = ts + MonthEnd(0)
            return (base + pd.Timedelta(days=int(spec["fallback_release_lag_calendar_days"]))).normalize()


        def build_release_aware_series(series_spec, trading_calendar):
            series_frame = fetch_fred_public_history(series_spec["series_id"])
            rows = []
            for _, row in series_frame.iterrows():
                rows.append(
                    {
                        "release_date": infer_release_date(row["observation_date"], series_spec),
                        "value": float(row["value"]),
                    }
                )
            releases = pd.DataFrame(rows).sort_values("release_date").drop_duplicates("release_date", keep="last")
            daily = pd.DataFrame(index=pd.DatetimeIndex(trading_calendar))
            daily = daily.join(releases.set_index("release_date"), how="left")
            daily["value"] = daily["value"].ffill()
            daily = daily.dropna().reset_index().rename(columns={"index": "Date"})
            series_id = series_spec["series_id"]
            daily[f"{series_id}_level"] = daily["value"]
            daily[f"{series_id}_delta"] = daily[f"{series_id}_level"].diff()
            rolling_mean = daily[f"{series_id}_level"].rolling(252, min_periods=60).mean()
            rolling_std = daily[f"{series_id}_level"].rolling(252, min_periods=60).std(ddof=0)
            daily[f"{series_id}_z"] = (daily[f"{series_id}_level"] - rolling_mean) / rolling_std.replace(0.0, np.nan)
            return daily.drop(columns=["value"]), releases.copy()


        price_df, volume_df = download_targets()
        trading_calendar = pd.DatetimeIndex(price_df.index).sort_values().unique()

        macro_daily = None
        release_lookup = {}
        for spec in macro_specs:
            frame, releases = build_release_aware_series(spec, trading_calendar)
            release_lookup[spec["series_id"]] = pd.DatetimeIndex(pd.to_datetime(releases["release_date"]).unique()).sort_values()
            if macro_daily is None:
                macro_daily = frame.copy()
            else:
                macro_daily = macro_daily.merge(frame, on="Date", how="outer")

        macro_daily = macro_daily.sort_values("Date").reset_index(drop=True)
        all_release_days = pd.DatetimeIndex(sorted({d for idx in release_lookup.values() for d in idx}))
        macro_daily["release_day_flag"] = macro_daily["Date"].isin(all_release_days).astype(int)
        important_release_map = {
            "days_since_cpi": "CPIAUCSL",
            "days_since_fomc": "FEDFUNDS",
            "days_since_nfp": "PAYEMS",
        }
        for feature_name, series_id in important_release_map.items():
            specific_release_days = release_lookup.get(series_id, pd.DatetimeIndex([]))
            values = []
            for current_date in pd.DatetimeIndex(macro_daily["Date"]):
                prior_dates = specific_release_days[specific_release_days <= current_date]
                if len(prior_dates) == 0:
                    values.append(np.nan)
                else:
                    values.append(int((current_date - prior_dates.max()).days))
            macro_daily[feature_name] = values
        """
    ),
    code(
        """
        log_return_df = np.log(price_df / price_df.shift(1))
        market_panel = pd.DataFrame(index=price_df.index)

        def rolling_max_drawdown(price_series, window=20):
            out = []
            values = price_series.astype(float).to_numpy()
            for idx in range(len(values)):
                start_idx = max(0, idx - window + 1)
                window_values = values[start_idx:idx + 1]
                running_max = np.maximum.accumulate(window_values)
                drawdowns = window_values / running_max - 1.0
                out.append(float(np.min(drawdowns)))
            return pd.Series(out, index=price_series.index)


        def downside_semivol(return_series, window=20):
            def _downside(x):
                x = np.asarray(x, dtype=float)
                negatives = x[x < 0]
                if len(negatives) == 0:
                    return 0.0
                return float(np.sqrt(np.mean(np.square(negatives))))
            return return_series.rolling(window, min_periods=window).apply(_downside, raw=True)


        for spec in target_specs:
            asset = spec["normalized_name"]
            ret_col = spec["target_column"]
            market_panel[f"{asset}_price"] = price_df[asset]
            market_panel[ret_col] = log_return_df[asset]
            market_panel[f"{asset}_return_5d"] = log_return_df[asset].rolling(5).sum()
            market_panel[f"{asset}_return_20d"] = log_return_df[asset].rolling(20).sum()
            market_panel[f"{asset}_realized_vol_5d"] = log_return_df[asset].rolling(5).std(ddof=0) * np.sqrt(252)
            market_panel[f"{asset}_realized_vol_20d"] = log_return_df[asset].rolling(20).std(ddof=0) * np.sqrt(252)
            market_panel[f"{asset}_downside_semivol_20d"] = downside_semivol(log_return_df[asset], 20) * np.sqrt(252)
            market_panel[f"{asset}_max_drawdown_20d"] = rolling_max_drawdown(price_df[asset], 20)
            market_panel[f"{asset}_rolling_skew_20d"] = log_return_df[asset].rolling(20).skew()
            market_panel[f"{asset}_rolling_kurtosis_20d"] = log_return_df[asset].rolling(20).kurt()
            if bool(spec.get("include_volume", False)) and volume_df is not None and asset in volume_df.columns:
                vol_series = volume_df[asset].astype(float)
                market_panel[f"{asset}_volume"] = vol_series
                mean_vol = vol_series.rolling(20, min_periods=20).mean()
                std_vol = vol_series.rolling(20, min_periods=20).std(ddof=0)
                market_panel[f"{asset}_volume_shock_20d"] = (vol_series - mean_vol) / std_vol.replace(0.0, np.nan)

        market_panel = market_panel.reset_index().rename(columns={"index": "Date"})
        market_panel["gold_minus_equity_return_5d"] = market_panel["GC_return_5d"] - market_panel["SPY_return_5d"]
        market_panel["oil_minus_equity_return_5d"] = market_panel["CL_return_5d"] - market_panel["SPY_return_5d"]
        market_panel["growth_minus_broad_equity_return_5d"] = market_panel["QQQ_return_5d"] - market_panel["SPY_return_5d"]
        market_panel["rates_minus_equity_return_5d"] = market_panel["ZN_return_5d"] - market_panel["SPY_return_5d"]

        return_columns = ["SPY_log_return", "QQQ_log_return", "GC_log_return", "CL_log_return", "ZN_log_return"]
        corr_vals = []
        for idx in range(len(market_panel)):
            start_idx = max(0, idx - 19)
            window = market_panel.iloc[start_idx:idx + 1][return_columns]
            if len(window) < 20 or window.isna().any().any():
                corr_vals.append(np.nan)
                continue
            corr = window.corr().to_numpy()
            tri = np.triu_indices_from(corr, k=1)
            corr_vals.append(float(np.nanmean(corr[tri])))
        market_panel["cross_asset_corr_20d"] = corr_vals
        market_panel["equity_realized_vol_20d"] = market_panel[["SPY_realized_vol_20d", "QQQ_realized_vol_20d"]].mean(axis=1)
        market_panel["commodity_realized_vol_20d"] = market_panel[["GC_realized_vol_20d", "CL_realized_vol_20d"]].mean(axis=1)
        market_panel["rates_realized_vol_20d"] = market_panel[["ZN_realized_vol_20d"]].mean(axis=1)

        aligned_live = market_panel.merge(macro_daily, on="Date", how="inner").sort_values("Date").dropna().reset_index(drop=True)


        def align_condition_columns(frame):
            out = frame.copy()
            missing_cols = [col for col in condition_cols if col not in out.columns]
            if missing_cols:
                print("Filling missing live condition columns from training means:", missing_cols)
                for col in missing_cols:
                    out[col] = condition_mean_map[col]
            return out
        """
    ),
    md(
        """
        ## Apply Scenario Shocks
        """
    ),
    code(
        """
        def apply_shocks_to_live_panel(aligned_df, scenario_rows):
            if scenario_rows.empty:
                return aligned_df.copy()

            out = aligned_df.copy()
            max_duration = int(scenario_rows["shock_duration_days"].fillna(model_config["seq_len"]).max())
            tail_start = max(0, len(out) - max(max_duration, int(model_config["seq_len"])))

            for _, row in scenario_rows.iterrows():
                series_id = str(row["series_id"]).strip()
                mode = str(row["shock_mode"]).strip().lower()
                value = float(row["shock_value"])
                if not series_id or mode == "none":
                    continue

                level_col = f"{series_id}_level"
                delta_col = f"{series_id}_delta"
                z_col = f"{series_id}_z"
                if level_col not in out.columns:
                    continue

                if mode == "delta":
                    out.loc[tail_start:, level_col] = out.loc[tail_start:, level_col] + value
                elif mode == "percent":
                    out.loc[tail_start:, level_col] = out.loc[tail_start:, level_col] * (1.0 + value)
                elif mode == "level":
                    out.loc[tail_start:, level_col] = value

                if delta_col in out.columns:
                    out[delta_col] = out[level_col].diff()
                if z_col in out.columns:
                    rolling_mean = out[level_col].rolling(252, min_periods=60).mean()
                    rolling_std = out[level_col].rolling(252, min_periods=60).std(ddof=0)
                    out[z_col] = (out[level_col] - rolling_mean) / rolling_std.replace(0.0, np.nan)

            return out


        scenario_panels = {}
        for scenario_id, group in scenarios_df.groupby("scenario_id", sort=False):
            scenario_panels[scenario_id] = align_condition_columns(apply_shocks_to_live_panel(aligned_live, group))

        print("Scenarios:", list(scenario_panels.keys()))
        """
    ),
    md(
        """
        ## Generate Scenario Paths
        """
    ),
    code(
        """
        def scale_condition_window(raw_window):
            return (raw_window - condition_mean.reshape(1, -1)) / condition_scale.reshape(1, -1)


        def generate_paths_for_condition_window(scaled_condition_window, latest_prices):
            cond_np = np.repeat(
                scaled_condition_window[None, :, :].astype(np.float32),
                RUN_CONFIG["generated_paths"],
                axis=0,
            )
            cond_tensor = torch.tensor(cond_np, dtype=torch.float32, device=RUN_CONFIG["device"])
            sample = scheduler.sample_reverse(
                model,
                cond_tensor,
                shape=(cond_tensor.shape[0], int(model_config["seq_len"]), int(model_config["input_dim"])),
            ).detach().cpu().numpy()
            unscaled = sample * return_scale.reshape(1, 1, -1) + return_mean.reshape(1, 1, -1)

            paths = {}
            horizon = min(RUN_CONFIG["forecast_horizon"], unscaled.shape[1])
            for idx, target_col in enumerate(target_cols):
                asset_name = target_col.replace("_log_return", "")
                start_price = float(latest_prices[asset_name])
                cumulative = np.cumsum(unscaled[:, :horizon, idx], axis=1)
                paths[asset_name] = start_price * np.exp(cumulative)
            return unscaled, paths


        latest_prices = {spec["normalized_name"]: float(price_df[spec["normalized_name"]].iloc[-1]) for spec in target_specs}
        scenario_outputs = {}
        scenario_summary_rows = []

        for scenario_id, scenario_frame in scenario_panels.items():
            condition_frame = scenario_frame[condition_cols].tail(int(model_config["seq_len"])).copy()
            if len(condition_frame) < int(model_config["seq_len"]):
                raise RuntimeError(f"Scenario {scenario_id} does not have enough rows for seq_len.")

            scaled_condition = scale_condition_window(condition_frame.to_numpy(dtype=np.float64))
            generated_returns, generated_paths = generate_paths_for_condition_window(scaled_condition, latest_prices)
            scenario_outputs[scenario_id] = {
                "returns": generated_returns,
                "paths": generated_paths,
            }

            for asset_name, price_paths in generated_paths.items():
                terminal_returns = price_paths[:, -1] / latest_prices[asset_name] - 1.0
                scenario_summary_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_name": scenarios_df.loc[scenarios_df["scenario_id"] == scenario_id, "scenario_name"].iloc[0],
                        "asset": asset_name,
                        "median_terminal_return": float(np.median(terminal_returns)),
                        "mean_terminal_return": float(np.mean(terminal_returns)),
                        "downside_probability": float(np.mean(terminal_returns < 0.0)),
                        "var_5": float(np.quantile(terminal_returns, 0.05)),
                        "cvar_5": float(terminal_returns[terminal_returns <= np.quantile(terminal_returns, 0.05)].mean()),
                    }
                )

        scenario_summary = pd.DataFrame(scenario_summary_rows)
        summary_path = LIVE_OUTPUT_DIR / "live_counterfactual_summary.csv"
        scenario_summary.to_csv(summary_path, index=False)
        display(scenario_summary)
        """
    ),
    md(
        """
        ## Plots
        """
    ),
    code(
        """
        horizon = min(RUN_CONFIG["forecast_horizon"], int(model_config["seq_len"]))
        x_axis = np.arange(1, horizon + 1)

        for target_col in target_cols:
            asset_name = target_col.replace("_log_return", "")
            fig, ax = plt.subplots(figsize=(12, 6))
            for scenario_id, scenario_data in scenario_outputs.items():
                price_paths = scenario_data["paths"][asset_name]
                median_path = np.quantile(price_paths, 0.50, axis=0)
                lower_path = np.quantile(price_paths, 0.10, axis=0)
                upper_path = np.quantile(price_paths, 0.90, axis=0)
                label = scenarios_df.loc[scenarios_df["scenario_id"] == scenario_id, "scenario_name"].iloc[0]
                ax.plot(x_axis, median_path, linewidth=2.2, label=label)
                ax.fill_between(x_axis, lower_path, upper_path, alpha=0.12)
            ax.axhline(latest_prices[asset_name], linestyle="--", linewidth=1.2, color="black", alpha=0.6)
            ax.set_title(f"{asset_name} live counterfactual scenarios")
            ax.set_xlabel("Forecast step")
            ax.set_ylabel("Price")
            ax.legend(loc="best")
            plt.tight_layout()
            fig.savefig(LIVE_OUTPUT_DIR / f"{asset_name.lower()}_live_counterfactual.png", dpi=180, bbox_inches="tight")
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
        print("Live counterfactual generation complete.")
        print("Summary table:", summary_path)
        print("Output directory:", LIVE_OUTPUT_DIR)
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
