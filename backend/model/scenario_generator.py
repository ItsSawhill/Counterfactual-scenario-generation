from __future__ import annotations

from typing import Any

from backend.model.generator_registry import get_generator, smoke_ddpm_enabled
from backend.model.schemas import ScenarioParameters


def _coerce_parameters(scenario_parameters: dict[str, Any] | ScenarioParameters | None) -> ScenarioParameters:
    if isinstance(scenario_parameters, ScenarioParameters):
        return scenario_parameters
    return ScenarioParameters(**(scenario_parameters or {}))


def generate_scenario(
    user_request: str,
    scenario_parameters: dict[str, Any] | ScenarioParameters,
    retrieved_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    params = _coerce_parameters(scenario_parameters)
    generator = get_generator()
    return generator.generate(
        inflation=params.inflation,
        interest_rate=params.interest_rate,
        horizon=params.horizon,
        path_count=params.path_count,
        user_request=user_request,
        retrieved_context=retrieved_context,
    )


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
