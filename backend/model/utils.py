from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import numpy as np


ASSETS = ["SPY", "QQQ", "GC", "CL", "ZN"]
START_PRICES = np.array([525.0, 452.0, 214.0, 78.0, 110.0], dtype=float)
BASE_DRIFT = np.array([0.00035, 0.00045, 0.00018, 0.00025, 0.00012], dtype=float)
BASE_VOL = np.array([0.010, 0.013, 0.011, 0.018, 0.006], dtype=float)
RETURN_SCALER_MEAN = np.zeros(len(ASSETS), dtype=float)
RETURN_SCALER_SCALE = np.ones(len(ASSETS), dtype=float)


def deterministic_seed(user_request: str, inflation: float, interest_rate: float, horizon: int, path_count: int) -> int:
    seed_material = f"{user_request}|{inflation:.6f}|{interest_rate:.6f}|{horizon}|{path_count}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def build_live_state_meta() -> dict[str, str | int]:
    now = datetime.now(UTC).replace(microsecond=0)
    timestamp = now.isoformat().replace("+00:00", "Z")
    as_of_date = now.date().isoformat()
    return {
        "market_data_as_of": as_of_date,
        "model_input_as_of": as_of_date,
        "live_fetched_at": timestamp,
        "cache_status": "fresh",
        "cache_age_seconds": 0,
        "cache_ttl_seconds": 900,
    }


def scenario_regime_adjustments(inflation: float, interest_rate: float) -> tuple[np.ndarray, np.ndarray]:
    inflation_gap = inflation - 0.03
    rate_gap = interest_rate - 0.04

    drift_adjustment = np.array(
        [
            -0.12 * inflation_gap - 0.10 * rate_gap,
            -0.18 * inflation_gap - 0.16 * rate_gap,
            0.08 * inflation_gap - 0.02 * rate_gap,
            -0.04 * inflation_gap - 0.06 * rate_gap,
            0.03 * inflation_gap - 0.08 * rate_gap,
        ],
        dtype=float,
    )

    vol_multiplier = 1.0 + np.array(
        [
            4.0 * abs(inflation_gap) + 2.0 * abs(rate_gap),
            4.5 * abs(inflation_gap) + 2.5 * abs(rate_gap),
            3.0 * abs(inflation_gap) + 1.2 * abs(rate_gap),
            5.0 * abs(inflation_gap) + 2.0 * abs(rate_gap),
            2.0 * abs(inflation_gap) + 2.5 * abs(rate_gap),
        ],
        dtype=float,
    )
    return drift_adjustment, vol_multiplier


def correlation_cholesky() -> np.ndarray:
    correlation = np.array(
        [
            [1.00, 0.82, -0.10, 0.45, -0.32],
            [0.82, 1.00, -0.08, 0.38, -0.36],
            [-0.10, -0.08, 1.00, 0.16, 0.12],
            [0.45, 0.38, 0.16, 1.00, -0.20],
            [-0.32, -0.36, 0.12, -0.20, 1.00],
        ],
        dtype=float,
    )
    return np.linalg.cholesky(correlation)
