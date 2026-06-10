from __future__ import annotations

from typing import Any

import numpy as np

from backend.model.schemas import ScenarioGenerationResponse, ScenarioParameters
from backend.model.utils import (
    ASSETS,
    BASE_DRIFT,
    BASE_VOL,
    RETURN_SCALER_MEAN,
    RETURN_SCALER_SCALE,
    START_PRICES,
    build_live_state_meta,
    correlation_cholesky,
    deterministic_seed,
    scenario_regime_adjustments,
)


def _coerce_parameters(scenario_parameters: dict[str, Any] | ScenarioParameters | None) -> ScenarioParameters:
    if isinstance(scenario_parameters, ScenarioParameters):
        return scenario_parameters
    return ScenarioParameters(**(scenario_parameters or {}))


def _context_signal(retrieved_context: list[dict[str, Any]] | None) -> float:
    if not retrieved_context:
        return 0.0
    text = " ".join(str(item.get("text", "")) for item in retrieved_context).lower()
    positive_terms = ("disinflation", "easing", "soft landing", "growth")
    negative_terms = ("stagflation", "hawkish", "reacceleration", "stress", "drawdown")
    score = sum(term in text for term in positive_terms) - sum(term in text for term in negative_terms)
    return float(np.clip(score * 0.00008, -0.00035, 0.00035))


def _generate_return_paths(params: ScenarioParameters, user_request: str, retrieved_context: list[dict[str, Any]] | None) -> np.ndarray:
    seed = deterministic_seed(
        user_request=user_request,
        inflation=params.inflation,
        interest_rate=params.interest_rate,
        horizon=params.horizon,
        path_count=params.path_count,
    )
    rng = np.random.default_rng(seed)

    drift_adjustment, vol_multiplier = scenario_regime_adjustments(params.inflation, params.interest_rate)
    context_adjustment = _context_signal(retrieved_context)
    asset_drift = BASE_DRIFT + drift_adjustment + context_adjustment
    asset_vol = BASE_VOL * vol_multiplier

    chol = correlation_cholesky()
    shocks = rng.normal(size=(params.path_count, params.horizon, len(ASSETS)))
    correlated = shocks @ chol.T

    # Fallback generator:
    # This approximates realistic cross-asset returns for UI and API integration.
    # TODO: Replace this block with extracted DDPM inference from
    # live_counterfactual_lab_v1.ipynb.
    horizon_profile = np.linspace(0.95, 1.05, params.horizon, dtype=float)[None, :, None]
    drift = asset_drift[None, None, :] * horizon_profile
    vol = asset_vol[None, None, :]
    return drift + correlated * vol


def generate_scenario(
    user_request: str,
    scenario_parameters: dict[str, Any] | ScenarioParameters,
    retrieved_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params = _coerce_parameters(scenario_parameters)
    returns = _generate_return_paths(params, user_request, retrieved_context)
    cumulative_prices = START_PRICES[None, None, :] * np.cumprod(1.0 + returns, axis=1)

    response = ScenarioGenerationResponse(
        assets=list(ASSETS),
        paths=returns.tolist(),
        return_scaler_mean=RETURN_SCALER_MEAN.tolist(),
        return_scaler_scale=RETURN_SCALER_SCALE.tolist(),
        start_prices=START_PRICES.tolist(),
        horizon=params.horizon,
        path_count=params.path_count,
        model_horizon=max(params.horizon, 30),
        live_state=build_live_state_meta(),
        metadata={
            "generator_type": "fallback_simulation",
            "ddpm_enabled": False,
            "notes": [
                "This response currently uses fallback stochastic logic shaped to the frontend contract.",
                "Notebook-based DDPM inference should replace the return generation block in backend/model/scenario_generator.py.",
            ],
            "price_path_preview_shape": list(cumulative_prices.shape),
            "context_count": len(retrieved_context or []),
        },
        fallback_generator_used=True,
        generator_type="fallback_simulation",
        ddpm_enabled=False,
    )
    return response.model_dump()


def simulate_frontend_request(inflation: float, interest_rate: float, horizon: int, path_count: int) -> dict[str, Any]:
    return generate_scenario(
        user_request="Frontend simulation request",
        scenario_parameters={
            "inflation": inflation,
            "interest_rate": interest_rate,
            "horizon": horizon,
            "path_count": path_count,
        },
        retrieved_context=None,
    )
