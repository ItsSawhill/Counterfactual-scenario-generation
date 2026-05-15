from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScenarioParameters(BaseModel):
    """Shared parameter model for backend requests.

    Assumptions:
    - The frontend currently sends snake_case keys to /simulate.
    - Internal code may also use camelCase names from TypeScript examples.
    """

    model_config = ConfigDict(populate_by_name=True)

    inflation: float = Field(0.03, ge=0.0, le=0.15)
    interest_rate: float = Field(0.04, ge=0.0, le=0.15, alias="interestRate")
    horizon: int = Field(30, ge=1, le=252)
    path_count: int = Field(128, ge=8, le=1024, alias="pathCount")


class ScenarioRequest(BaseModel):
    user_request: str = Field(..., min_length=3)
    scenario_parameters: ScenarioParameters = Field(default_factory=ScenarioParameters)
    retrieved_context: list[dict[str, Any]] | None = None


class ScenarioPath(BaseModel):
    asset: str
    price_paths: list[list[float]]


class LiveStateMeta(BaseModel):
    market_data_as_of: str
    model_input_as_of: str
    live_fetched_at: str
    cache_status: str
    cache_age_seconds: int
    cache_ttl_seconds: int


class ScenarioGenerationResponse(BaseModel):
    assets: list[str]
    paths: list[list[list[float]]]
    return_scaler_mean: list[float]
    return_scaler_scale: list[float]
    start_prices: list[float]
    horizon: int
    path_count: int
    model_horizon: int
    live_state: LiveStateMeta
    metadata: dict[str, Any]
    fallback_generator_used: bool
