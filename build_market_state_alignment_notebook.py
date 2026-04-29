from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("market_state_and_alignment_builder.ipynb")


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
        # Market State And Alignment Builder

        This notebook builds the market-side half of the upgraded counterfactual dataset and merges it with the release-aware macro output.

        ## What it does

        - downloads the upgraded target universe from Yahoo Finance
        - computes daily log returns
        - computes rolling market-state features
        - computes cross-asset spreads and rolling correlation summaries
        - merges the market panel with the release-aware macro panel
        - writes the aligned dataset used for the next DDPM retraining run

        ## Important note on futures

        Yahoo's futures tickers are a practical prototype source, but they are not a perfect institutional continuous-contract data feed. In this notebook they are treated as provisional continuous proxies so we can move the pipeline forward. If you later switch to a proper back-adjusted futures source, the schema can stay the same while the data backend improves.
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
        import yfinance as yf
        from IPython.display import display

        plt.style.use("seaborn-v0_8-darkgrid")
        pd.set_option("display.max_columns", 160)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        """
    ),
    md(
        """
        ## Configuration

        This notebook reads the target universe and output locations from `dataset_spec_v1.json`, then reads the release-aware macro output produced by the macro loader notebook.
        """
    ),
    code(
        """
        SPEC_PATH = Path("dataset_spec_v1.json")
        ARTIFACT_ROOT = Path.cwd() / "counterfactual_data_build"

        with open(SPEC_PATH, "r", encoding="utf-8") as handle:
            dataset_spec = json.load(handle)

        target_specs = dataset_spec["targets"]
        macro_output_relative = Path(dataset_spec["loader_outputs"]["macro_release_aware_daily_csv"])
        market_output_relative = Path(dataset_spec["loader_outputs"]["market_panel_csv"])
        aligned_output_relative = Path(dataset_spec["loader_outputs"]["aligned_dataset_csv"])

        macro_path = ARTIFACT_ROOT / macro_output_relative
        market_output_path = ARTIFACT_ROOT / market_output_relative
        aligned_output_path = ARTIFACT_ROOT / aligned_output_relative
        metadata_output_path = ARTIFACT_ROOT / "processed" / "market" / "market_builder_metadata.csv"

        market_output_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_output_path.parent.mkdir(parents=True, exist_ok=True)

        if not macro_path.exists():
            raise FileNotFoundError(
                f"Missing release-aware macro file: {macro_path}. Run release_aware_macro_loader.ipynb first."
            )

        macro_daily = pd.read_csv(macro_path)
        macro_daily["Date"] = pd.to_datetime(macro_daily["Date"]).dt.normalize()

        display(pd.DataFrame(target_specs)[["symbol", "normalized_name", "asset_class", "target_column"]])
        print("Macro input:", macro_path)
        print("Market output:", market_output_path)
        print("Aligned output:", aligned_output_path)
        """
    ),
    md(
        """
        ## Download Market Data

        This section downloads prices and volume for the upgraded target universe and normalizes column names so later feature engineering stays readable.
        """
    ),
    code(
        """
        def choose_price_frame(raw_download):
            if raw_download is None or len(raw_download) == 0:
                raise RuntimeError("No market data returned from yfinance.")

            if isinstance(raw_download.columns, pd.MultiIndex):
                level0 = list(raw_download.columns.get_level_values(0).unique())
                if "Adj Close" in level0:
                    prices = raw_download["Adj Close"].copy()
                elif "Close" in level0:
                    prices = raw_download["Close"].copy()
                else:
                    raise RuntimeError(f"Unexpected price columns: {level0}")

                volume = raw_download["Volume"].copy() if "Volume" in level0 else None
            else:
                price_column = "Adj Close" if "Adj Close" in raw_download.columns else "Close"
                ticker_name = raw_download.attrs.get("ticker", "asset")
                prices = raw_download[[price_column]].copy()
                prices.columns = [ticker_name]
                volume = raw_download[["Volume"]].copy() if "Volume" in raw_download.columns else None
                if volume is not None:
                    volume.columns = [ticker_name]

            prices.index = pd.to_datetime(prices.index).normalize()
            prices = prices.sort_index().ffill()

            if volume is not None:
                volume.index = pd.to_datetime(volume.index).normalize()
                volume = volume.sort_index()

            return prices, volume


        def download_target_history(target_specs, start_date, end_date):
            tickers = [item["symbol"] for item in target_specs]
            raw = yf.download(
                tickers,
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            prices, volumes = choose_price_frame(raw)
            rename_map = {
                spec["symbol"]: spec["normalized_name"]
                for spec in target_specs
            }

            prices = prices.rename(columns=rename_map)
            if volumes is not None:
                volumes = volumes.rename(columns=rename_map)

            return prices, volumes


        calendar_spec = dataset_spec["calendar"]
        start_date = calendar_spec["start_date"]
        end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

        price_df, volume_df = download_target_history(target_specs, start_date, end_date)
        price_df = price_df.dropna(how="all").ffill()

        print("Price frame shape:", price_df.shape)
        display(price_df.tail())
        """
    ),
    md(
        """
        ## Market Feature Engineering

        These are the first market-state features we want the next DDPM retraining run to see:

        - 1d / 5d / 20d returns
        - realized vol and downside semivol
        - drawdown
        - rolling skew and kurtosis
        - optional volume shock for ETF targets
        """
    ),
    code(
        """
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


        log_return_df = np.log(price_df / price_df.shift(1))
        market_panel = pd.DataFrame(index=price_df.index)

        metadata_rows = []

        for spec in target_specs:
            asset = spec["normalized_name"]
            asset_class = spec["asset_class"]
            include_volume = bool(spec.get("include_volume", False))
            price_col = asset
            return_col = spec["target_column"]

            market_panel[f"{asset}_price"] = price_df[price_col]
            market_panel[return_col] = log_return_df[price_col]
            market_panel[f"{asset}_return_5d"] = log_return_df[price_col].rolling(5).sum()
            market_panel[f"{asset}_return_20d"] = log_return_df[price_col].rolling(20).sum()
            market_panel[f"{asset}_realized_vol_5d"] = log_return_df[price_col].rolling(5).std(ddof=0) * np.sqrt(252)
            market_panel[f"{asset}_realized_vol_20d"] = log_return_df[price_col].rolling(20).std(ddof=0) * np.sqrt(252)
            market_panel[f"{asset}_downside_semivol_20d"] = downside_semivol(log_return_df[price_col], window=20) * np.sqrt(252)
            market_panel[f"{asset}_max_drawdown_20d"] = rolling_max_drawdown(price_df[price_col], window=20)
            market_panel[f"{asset}_rolling_skew_20d"] = log_return_df[price_col].rolling(20).skew()
            market_panel[f"{asset}_rolling_kurtosis_20d"] = log_return_df[price_col].rolling(20).kurt()

            if include_volume and volume_df is not None and asset in volume_df.columns:
                vol_series = volume_df[asset].astype(float)
                market_panel[f"{asset}_volume"] = vol_series
                rolling_vol_mean = vol_series.rolling(20, min_periods=20).mean()
                rolling_vol_std = vol_series.rolling(20, min_periods=20).std(ddof=0)
                market_panel[f"{asset}_volume_shock_20d"] = (vol_series - rolling_vol_mean) / rolling_vol_std.replace(0.0, np.nan)

            metadata_rows.append(
                {
                    "asset": asset,
                    "symbol": spec["symbol"],
                    "asset_class": asset_class,
                    "include_volume": include_volume,
                    "continuous_contract_required": bool(spec.get("continuous_contract_required", False)),
                }
            )

        market_panel = market_panel.reset_index().rename(columns={"index": "Date"})
        market_panel["Date"] = pd.to_datetime(market_panel["Date"]).dt.normalize()

        display(market_panel.tail())
        """
    ),
    md(
        """
        ## Cross-Asset Features

        These summarize relative moves and co-movement regimes rather than only individual asset behavior.
        """
    ),
    code(
        """
        core_assets = {spec["normalized_name"] for spec in target_specs}

        required_assets = {"SPY", "QQQ", "GC", "CL", "ZN"}
        missing_assets = required_assets - core_assets
        if missing_assets:
            raise ValueError(f"Target universe missing required assets for cross-asset features: {missing_assets}")

        market_panel = market_panel.sort_values("Date").reset_index(drop=True)
        market_panel["gold_minus_equity_return_5d"] = market_panel["GC_return_5d"] - market_panel["SPY_return_5d"]
        market_panel["oil_minus_equity_return_5d"] = market_panel["CL_return_5d"] - market_panel["SPY_return_5d"]
        market_panel["growth_minus_broad_equity_return_5d"] = market_panel["QQQ_return_5d"] - market_panel["SPY_return_5d"]
        market_panel["rates_minus_equity_return_5d"] = market_panel["ZN_return_5d"] - market_panel["SPY_return_5d"]

        return_columns = [
            "SPY_log_return",
            "QQQ_log_return",
            "GC_log_return",
            "CL_log_return",
            "ZN_log_return",
        ]

        cross_asset_corr = []
        returns_only = market_panel[return_columns]
        for idx in range(len(market_panel)):
            start_idx = max(0, idx - 19)
            window = returns_only.iloc[start_idx:idx + 1]
            if len(window) < 20 or window.isna().any().any():
                cross_asset_corr.append(np.nan)
                continue
            corr_matrix = window.corr().to_numpy(dtype=float)
            upper_idx = np.triu_indices_from(corr_matrix, k=1)
            cross_asset_corr.append(float(np.nanmean(corr_matrix[upper_idx])))

        market_panel["cross_asset_corr_20d"] = cross_asset_corr

        equity_vol_cols = ["SPY_realized_vol_20d", "QQQ_realized_vol_20d"]
        commodity_vol_cols = ["GC_realized_vol_20d", "CL_realized_vol_20d"]
        rates_vol_cols = ["ZN_realized_vol_20d"]

        market_panel["equity_realized_vol_20d"] = market_panel[equity_vol_cols].mean(axis=1)
        market_panel["commodity_realized_vol_20d"] = market_panel[commodity_vol_cols].mean(axis=1)
        market_panel["rates_realized_vol_20d"] = market_panel[rates_vol_cols].mean(axis=1)

        display(market_panel[[
            "Date",
            "gold_minus_equity_return_5d",
            "oil_minus_equity_return_5d",
            "growth_minus_broad_equity_return_5d",
            "rates_minus_equity_return_5d",
            "cross_asset_corr_20d",
        ]].tail())
        """
    ),
    md(
        """
        ## Merge With Release-Aware Macro Panel

        The result should be the first training-ready aligned dataset for the upgraded build.
        """
    ),
    code(
        """
        aligned_df = market_panel.merge(macro_daily, on="Date", how="inner")
        aligned_df = aligned_df.sort_values("Date").reset_index(drop=True)

        before_dropna_rows = len(aligned_df)
        aligned_df = aligned_df.dropna().reset_index(drop=True)
        after_dropna_rows = len(aligned_df)

        print("Aligned rows before dropna:", before_dropna_rows)
        print("Aligned rows after dropna:", after_dropna_rows)
        display(aligned_df.tail())
        """
    ),
    md(
        """
        ## Save Outputs
        """
    ),
    code(
        """
        market_metadata = pd.DataFrame(metadata_rows)
        market_panel.to_csv(market_output_path, index=False)
        aligned_df.to_csv(aligned_output_path, index=False)
        market_metadata.to_csv(metadata_output_path, index=False)

        print("Saved:")
        print(market_output_path)
        print(aligned_output_path)
        print(metadata_output_path)
        """
    ),
    md(
        """
        ## Diagnostics

        These quick charts make it easy to sanity-check that the upgraded universe and derived features look stable enough before we push them into a training notebook.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        price_cols = [f"{spec['normalized_name']}_price" for spec in target_specs]
        aligned_df.plot(x="Date", y=price_cols, ax=axes[0], linewidth=1.5)
        axes[0].set_title("Target price history")
        axes[0].set_ylabel("Price")

        aligned_df.plot(
            x="Date",
            y=["equity_realized_vol_20d", "commodity_realized_vol_20d", "rates_realized_vol_20d"],
            ax=axes[1],
            linewidth=1.5,
        )
        axes[1].set_title("Grouped realized volatility features")
        axes[1].set_ylabel("Annualized vol")

        plt.tight_layout()
        plt.show()
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
