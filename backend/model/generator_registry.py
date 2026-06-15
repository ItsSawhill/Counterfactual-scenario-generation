from __future__ import annotations

import os
from typing import Any, Protocol

import numpy as np

from backend.model.schemas import ScenarioGenerationResponse, ScenarioParameters
from backend.model.smoke_inference import (
    generate_smoke_ddpm_response,
    smoke_artifact_status,
)
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


GENERATOR_MODE_ENV = "GENERATOR_MODE"
SMOKE_DDPM_ENABLED_ENV = "SMOKE_DDPM_ENABLED"
FALLBACK_MODE = "fallback"
SMOKE_DDPM_MODE = "smoke_ddpm"
VALID_GENERATOR_MODES = (FALLBACK_MODE, SMOKE_DDPM_MODE)


class ScenarioGenerator(Protocol):
    mode: str
    generator_type: str
    ddpm_enabled: bool

    def load(self) -> None:
        """Prepare any required model artifacts for inference."""

    def generate(
        self,
        *,
        inflation: float,
        interest_rate: float,
        horizon: int,
        path_count: int,
        user_request: str = "Scenario generation request",
        retrieved_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the standardized ScenarioGenerationResponse payload."""

    def artifact_status(self) -> dict[str, Any]:
        """Return artifact readiness details for startup logging."""


def smoke_ddpm_enabled() -> bool:
    return os.getenv(SMOKE_DDPM_ENABLED_ENV, "").strip() == "1"


def _normalize_generator_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or "").strip().lower()
    if not mode:
        return SMOKE_DDPM_MODE if smoke_ddpm_enabled() else FALLBACK_MODE
    if mode not in VALID_GENERATOR_MODES:
        expected = ", ".join(VALID_GENERATOR_MODES)
        raise RuntimeError(
            f"Invalid {GENERATOR_MODE_ENV}={raw_mode!r}. Expected one of: {expected}."
        )
    return mode


def _context_signal(retrieved_context: list[dict[str, Any]] | None) -> float:
    if not retrieved_context:
        return 0.0
    text = " ".join(str(item.get("text", "")) for item in retrieved_context).lower()
    positive_terms = ("disinflation", "easing", "soft landing", "growth")
    negative_terms = ("stagflation", "hawkish", "reacceleration", "stress", "drawdown")
    score = sum(term in text for term in positive_terms) - sum(term in text for term in negative_terms)
    return float(np.clip(score * 0.00008, -0.00035, 0.00035))


class FallbackSimulationGenerator:
    mode = FALLBACK_MODE
    generator_type = "fallback_simulation"
    ddpm_enabled = False

    def load(self) -> None:
        return None

    def artifact_status(self) -> dict[str, Any]:
        return {"required": False, "ready": True}

    def _generate_return_paths(
        self,
        *,
        params: ScenarioParameters,
        user_request: str,
        retrieved_context: list[dict[str, Any]] | None,
    ) -> np.ndarray:
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

        horizon_profile = np.linspace(0.95, 1.05, params.horizon, dtype=float)[None, :, None]
        drift = asset_drift[None, None, :] * horizon_profile
        vol = asset_vol[None, None, :]
        return drift + correlated * vol

    def generate(
        self,
        *,
        inflation: float,
        interest_rate: float,
        horizon: int,
        path_count: int,
        user_request: str = "Scenario generation request",
        retrieved_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        params = ScenarioParameters(
            inflation=inflation,
            interest_rate=interest_rate,
            horizon=horizon,
            path_count=path_count,
        )
        returns = self._generate_return_paths(
            params=params,
            user_request=user_request,
            retrieved_context=retrieved_context,
        )
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
                "generator_type": self.generator_type,
                "ddpm_enabled": self.ddpm_enabled,
                "notes": [
                    "This response currently uses fallback stochastic logic shaped to the frontend contract.",
                    "Future DDPM engines should plug into backend/model/generator_registry.py.",
                ],
                "price_path_preview_shape": list(cumulative_prices.shape),
                "context_count": len(retrieved_context or []),
            },
            fallback_generator_used=True,
            generator_type=self.generator_type,
            ddpm_enabled=self.ddpm_enabled,
        )
        return response.model_dump()


class SmokeDDPMGenerator:
    mode = SMOKE_DDPM_MODE
    generator_type = "smoke_ddpm"
    ddpm_enabled = True

    def load(self) -> None:
        status = smoke_artifact_status()
        missing = [name for name, ready in status.items() if not ready]
        if missing:
            raise RuntimeError(
                "Smoke DDPM artifacts are missing. Run `python run_counterfactual_pipeline.py --smoke` first. "
                f"Missing: {', '.join(missing)}"
            )

    def artifact_status(self) -> dict[str, Any]:
        status = smoke_artifact_status()
        return {**status, "ready": all(status.values())}

    def generate(
        self,
        *,
        inflation: float,
        interest_rate: float,
        horizon: int,
        path_count: int,
        user_request: str = "Scenario generation request",
        retrieved_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _ = (inflation, interest_rate, user_request, retrieved_context)
        return generate_smoke_ddpm_response(path_count=path_count, horizon=horizon)


def get_generator(mode: str | None = None) -> ScenarioGenerator:
    selected_mode = _normalize_generator_mode(mode if mode is not None else os.getenv(GENERATOR_MODE_ENV))
    if selected_mode == FALLBACK_MODE:
        return FallbackSimulationGenerator()
    if selected_mode == SMOKE_DDPM_MODE:
        return SmokeDDPMGenerator()
    raise RuntimeError(f"Unsupported generator mode after normalization: {selected_mode}")


def get_generator_runtime_status(mode: str | None = None) -> dict[str, Any]:
    generator = get_generator(mode)
    return {
        "selected_generator": generator.generator_type,
        "mode": generator.mode,
        "artifact_status": generator.artifact_status(),
        "ddpm_enabled": generator.ddpm_enabled,
    }
