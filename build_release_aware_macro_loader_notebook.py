from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("release_aware_macro_loader.ipynb")


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
        # Release-Aware Macro Loader

        This notebook builds the macro portion of the upgraded counterfactual dataset using release-aware timing. The goal is to stop leaking hindsight-clean macro values into the training set.

        ## What it produces

        - a release event table for each macro series
        - a daily macro panel aligned to the trading calendar
        - daily features that reflect only what was known on each date

        ## Data modes

        The notebook supports two paths:

        - ALFRED vintage mode when `FRED_API_KEY` is available
        - fallback release-lag mode when a key is not available

        Vintage mode is preferred because it uses real-time availability dates. The fallback mode is still useful for building and testing the pipeline, but it is an approximation.
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
        import os
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import requests
        import yfinance as yf
        from IPython.display import display
        from pandas.tseries.offsets import BDay, MonthEnd

        plt.style.use("seaborn-v0_8-darkgrid")
        pd.set_option("display.max_columns", 120)
        pd.set_option("display.width", 180)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        """
    ),
    md(
        """
        ## Configuration

        The notebook reads the canonical series definitions from `dataset_spec_v1.json`. You can point `ARTIFACT_ROOT` wherever you want the output CSVs to land.
        """
    ),
    code(
        """
        SPEC_PATH = Path("dataset_spec_v1.json")
        ARTIFACT_ROOT = Path.cwd() / "counterfactual_data_build"
        OUTPUT_ROOT = ARTIFACT_ROOT / "processed" / "macro"
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        with open(SPEC_PATH, "r", encoding="utf-8") as handle:
            dataset_spec = json.load(handle)

        FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
        USE_ALFRED = bool(FRED_API_KEY)

        print("Spec path:", SPEC_PATH.resolve())
        print("Artifact root:", ARTIFACT_ROOT.resolve())
        print("Using ALFRED vintages:", USE_ALFRED)
        """
    ),
    code(
        """
        macro_series_spec = dataset_spec["macro_series"]
        calendar_spec = dataset_spec["calendar"]

        calendar_start = calendar_spec["start_date"]
        calendar_end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        calendar_tickers = calendar_spec["calendar_anchor_tickers"]

        display(pd.DataFrame(macro_series_spec)[[
            "series_id",
            "display_name",
            "frequency",
            "family",
            "use_vintages",
            "fallback_release_lag_calendar_days",
        ]])
        """
    ),
    md(
        """
        ## Helper Functions

        These functions:

        - build the trading calendar
        - fetch public FRED history
        - optionally fetch ALFRED vintages
        - infer release dates
        - convert release events into a known-as-of daily panel
        """
    ),
    code(
        """
        def build_trading_calendar(tickers, start_date, end_date):
            raw = yf.download(
                tickers,
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if raw is None or len(raw) == 0:
                raise RuntimeError("Could not build trading calendar from yfinance.")

            if isinstance(raw.columns, pd.MultiIndex):
                if "Adj Close" in raw.columns.get_level_values(0):
                    closes = raw["Adj Close"].copy()
                else:
                    closes = raw["Close"].copy()
                calendar_index = pd.DatetimeIndex(closes.dropna(how="all").index).normalize()
            else:
                calendar_index = pd.DatetimeIndex(raw.index).normalize()

            return calendar_index.sort_values().unique()


        def fetch_fred_public_history(series_id):
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            frame = pd.read_csv(url)
            frame = frame.rename(columns={"DATE": "observation_date", series_id: "value"})
            frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.normalize()
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["value"]).sort_values("observation_date").reset_index(drop=True)
            return frame


        def fetch_alfred_vintages(series_id, api_key):
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "realtime_start": "1776-07-04",
                "realtime_end": "9999-12-31",
            }
            url = "https://api.stlouisfed.org/fred/series/observations"
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()

            rows = []
            for obs in payload.get("observations", []):
                value = obs.get("value")
                if value in (None, ".", ""):
                    continue
                rows.append(
                    {
                        "observation_date": pd.Timestamp(obs["date"]).normalize(),
                        "realtime_start": pd.Timestamp(obs["realtime_start"]).normalize(),
                        "realtime_end": pd.Timestamp(obs["realtime_end"]).normalize(),
                        "value": float(value),
                    }
                )

            if not rows:
                raise RuntimeError(f"No ALFRED data returned for {series_id}.")

            frame = pd.DataFrame(rows).drop_duplicates(
                subset=["observation_date", "realtime_start", "value"]
            )
            return frame.sort_values(["realtime_start", "observation_date"]).reset_index(drop=True)


        def infer_fallback_release_date(observation_date, series_spec):
            anchor = series_spec["observation_anchor"]
            lag_days = int(series_spec["fallback_release_lag_calendar_days"])

            ts = pd.Timestamp(observation_date).normalize()

            if anchor == "same_day":
                base_date = ts
            elif anchor == "month_end":
                base_date = ts + MonthEnd(0)
            else:
                base_date = ts

            return (base_date + pd.Timedelta(days=lag_days)).normalize()


        def build_release_events_from_vintages(vintage_frame, series_spec):
            series_id = series_spec["series_id"]
            release_rows = []

            for release_date, subframe in vintage_frame.groupby("realtime_start", sort=True):
                latest_obs = subframe.sort_values(["observation_date", "realtime_start"]).iloc[-1]
                release_rows.append(
                    {
                        "series_id": series_id,
                        "release_date": pd.Timestamp(release_date).normalize(),
                        "observation_date": pd.Timestamp(latest_obs["observation_date"]).normalize(),
                        "value": float(latest_obs["value"]),
                        "release_mode": "alfred_vintage",
                    }
                )

            out = pd.DataFrame(release_rows)
            out = out.sort_values(["series_id", "release_date", "observation_date"]).reset_index(drop=True)
            return out


        def build_release_events_fallback(series_frame, series_spec):
            release_rows = []
            for _, row in series_frame.iterrows():
                release_rows.append(
                    {
                        "series_id": series_spec["series_id"],
                        "release_date": infer_fallback_release_date(row["observation_date"], series_spec),
                        "observation_date": pd.Timestamp(row["observation_date"]).normalize(),
                        "value": float(row["value"]),
                        "release_mode": "fallback_release_lag",
                    }
                )
            out = pd.DataFrame(release_rows)
            out = out.sort_values(["series_id", "release_date", "observation_date"]).reset_index(drop=True)
            return out


        def release_events_to_daily_panel(release_events, trading_calendar):
            release_events = release_events.sort_values("release_date").reset_index(drop=True)
            working = release_events[["release_date", "value"]].drop_duplicates(subset=["release_date"], keep="last")
            working = working.rename(columns={"release_date": "Date"})
            working["Date"] = pd.to_datetime(working["Date"]).dt.normalize()
            working = working.set_index("Date").sort_index()

            daily = pd.DataFrame(index=pd.DatetimeIndex(trading_calendar))
            daily = daily.join(working[["value"]], how="left")
            daily["value"] = daily["value"].ffill()
            daily = daily.dropna().reset_index().rename(columns={"index": "Date"})
            return daily


        def add_macro_features(daily_frame, series_id):
            out = daily_frame.copy()
            value_col = f"{series_id}_level"
            delta_col = f"{series_id}_delta"
            z_col = f"{series_id}_z"

            out = out.rename(columns={"value": value_col})
            out[delta_col] = out[value_col].diff()
            rolling_mean = out[value_col].rolling(252, min_periods=60).mean()
            rolling_std = out[value_col].rolling(252, min_periods=60).std(ddof=0)
            out[z_col] = (out[value_col] - rolling_mean) / rolling_std.replace(0.0, np.nan)
            return out
        """
    ),
    md(
        """
        ## Build Trading Calendar
        """
    ),
    code(
        """
        trading_calendar = build_trading_calendar(calendar_tickers, calendar_start, calendar_end)
        print("Trading calendar length:", len(trading_calendar))
        print("Calendar start:", trading_calendar.min())
        print("Calendar end:", trading_calendar.max())
        """
    ),
    md(
        """
        ## Build Release Events And Daily Macro Panel

        For each macro series:

        - fetch public history
        - use ALFRED vintages when configured and available
        - otherwise infer release dates from the fallback lag rule
        - convert release events into a trading-day-aligned known-as-of series
        """
    ),
    code(
        """
        release_event_frames = []
        daily_macro_frames = []
        macro_metadata_rows = []

        for series_spec in macro_series_spec:
            series_id = series_spec["series_id"]
            public_history = fetch_fred_public_history(series_id)

            use_vintage_mode = bool(series_spec["use_vintages"]) and USE_ALFRED
            if use_vintage_mode:
                try:
                    vintage_frame = fetch_alfred_vintages(series_id, FRED_API_KEY)
                    release_events = build_release_events_from_vintages(vintage_frame, series_spec)
                    mode_used = "alfred_vintage"
                except Exception as exc:
                    print(f"{series_id}: ALFRED failed, falling back to lag approximation. Reason: {exc}")
                    release_events = build_release_events_fallback(public_history, series_spec)
                    mode_used = "fallback_release_lag"
            else:
                release_events = build_release_events_fallback(public_history, series_spec)
                mode_used = "fallback_release_lag"

            daily_panel = release_events_to_daily_panel(release_events, trading_calendar)
            daily_panel = add_macro_features(daily_panel, series_id)

            release_event_frames.append(release_events.assign(mode_used=mode_used))
            daily_macro_frames.append(daily_panel)

            macro_metadata_rows.append(
                {
                    "series_id": series_id,
                    "mode_used": mode_used,
                    "raw_observations": int(len(public_history)),
                    "release_events": int(len(release_events)),
                    "daily_rows": int(len(daily_panel)),
                    "first_release_date": str(pd.to_datetime(release_events["release_date"]).min().date()),
                    "last_release_date": str(pd.to_datetime(release_events["release_date"]).max().date()),
                }
            )

        release_events_all = pd.concat(release_event_frames, ignore_index=True)
        macro_metadata = pd.DataFrame(macro_metadata_rows)

        macro_daily = None
        for frame in daily_macro_frames:
            if macro_daily is None:
                macro_daily = frame.copy()
            else:
                macro_daily = macro_daily.merge(frame, on="Date", how="outer")

        macro_daily = macro_daily.sort_values("Date").reset_index(drop=True)
        display(macro_metadata)
        display(macro_daily.tail())
        """
    ),
    md(
        """
        ## Add Release Timing Features

        These features are useful later for the diffusion model because they tell the model whether it is close to a major information refresh.
        """
    ),
    code(
        """
        release_day_table = (
            release_events_all[["series_id", "release_date"]]
            .drop_duplicates()
            .assign(flag=1)
        )

        macro_daily_features = macro_daily.copy()
        macro_daily_features["release_day_flag"] = 0

        important_series_for_days_since = {
            "days_since_cpi": "CPIAUCSL",
            "days_since_fomc": "FEDFUNDS",
            "days_since_nfp": "PAYEMS",
        }

        all_release_days = pd.DatetimeIndex(pd.to_datetime(release_events_all["release_date"]).unique()).sort_values()
        macro_daily_features["release_day_flag"] = macro_daily_features["Date"].isin(all_release_days).astype(int)

        for feature_name, series_id in important_series_for_days_since.items():
            specific_release_days = pd.DatetimeIndex(
                pd.to_datetime(
                    release_events_all.loc[release_events_all["series_id"] == series_id, "release_date"]
                ).unique()
            ).sort_values()

            values = []
            for current_date in pd.DatetimeIndex(macro_daily_features["Date"]):
                prior_dates = specific_release_days[specific_release_days <= current_date]
                if len(prior_dates) == 0:
                    values.append(np.nan)
                else:
                    values.append(int((current_date - prior_dates.max()).days))
            macro_daily_features[feature_name] = values

        display(macro_daily_features.tail())
        """
    ),
    md(
        """
        ## Save Outputs
        """
    ),
    code(
        """
        release_events_path = OUTPUT_ROOT / "macro_release_events.csv"
        macro_daily_path = OUTPUT_ROOT / "macro_release_aware_daily.csv"
        macro_metadata_path = OUTPUT_ROOT / "macro_loader_metadata.csv"

        release_events_all.to_csv(release_events_path, index=False)
        macro_daily_features.to_csv(macro_daily_path, index=False)
        macro_metadata.to_csv(macro_metadata_path, index=False)

        print("Saved:")
        print(release_events_path)
        print(macro_daily_path)
        print(macro_metadata_path)
        """
    ),
    md(
        """
        ## Quick Diagnostic Plot

        This is just a quick sanity check to make sure the release-aware series is stair-stepping the way it should rather than drifting with future information.
        """
    ),
    code(
        """
        preview_cols = ["Date", "CPIAUCSL_level", "UNRATE_level", "FEDFUNDS_level"]
        preview = macro_daily_features[preview_cols].dropna().tail(750).copy()

        fig, ax = plt.subplots(figsize=(12, 5))
        for col in preview_cols[1:]:
            ax.plot(preview["Date"], preview[col], label=col)
        ax.set_title("Release-aware macro level preview")
        ax.set_xlabel("Date")
        ax.legend(loc="best")
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
