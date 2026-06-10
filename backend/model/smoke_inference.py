from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend.model.schemas import ScenarioGenerationResponse
from backend.model.smoke_ddpm import SmokeDenoiser, SmokeScheduler
from backend.model.utils import build_live_state_meta


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = PROJECT_ROOT / "counterfactual_data_build" / "smoke"


@dataclass(frozen=True)
class SmokeArtifactPaths:
    root: Path = SMOKE_ROOT

    @property
    def model_config(self) -> Path:
        return self.root / "outputs" / "tables" / "ddpm_smoke_model_config.json"

    @property
    def scalers(self) -> Path:
        return self.root / "processed" / "windows" / "scalers_smoke.json"

    @property
    def checkpoint(self) -> Path:
        return self.root / "outputs" / "checkpoints" / "conditional_ddpm_smoke_best.pt"

    @property
    def c_test(self) -> Path:
        return self.root / "processed" / "windows" / "C_test_smoke.npy"


class SmokeArtifactsMissingError(RuntimeError):
    pass


def smoke_artifact_status(paths: SmokeArtifactPaths | None = None) -> dict[str, bool]:
    paths = paths or SmokeArtifactPaths()
    return {
        "model_config": paths.model_config.exists(),
        "scalers": paths.scalers.exists(),
        "checkpoint": paths.checkpoint.exists(),
        "c_test": paths.c_test.exists(),
    }


def smoke_artifacts_ready(paths: SmokeArtifactPaths | None = None) -> bool:
    return all(smoke_artifact_status(paths).values())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_artifacts(paths: SmokeArtifactPaths) -> None:
    status = smoke_artifact_status(paths)
    missing = [name for name, exists in status.items() if not exists]
    if missing:
        details = ", ".join(f"{name}={getattr(paths, name)}" for name in missing)
        raise SmokeArtifactsMissingError(
            "Smoke DDPM artifacts are missing. Run `python run_counterfactual_pipeline.py --smoke` first. "
            f"Missing: {details}"
        )


def generate_smoke_ddpm_response(path_count: int, horizon: int, paths: SmokeArtifactPaths | None = None) -> dict[str, Any]:
    paths = paths or SmokeArtifactPaths()
    _require_artifacts(paths)

    model_config = _load_json(paths.model_config)
    scalers = _load_json(paths.scalers)
    c_test = np.load(paths.c_test).astype(np.float32)
    if c_test.ndim != 3 or len(c_test) == 0:
        raise RuntimeError(f"Smoke C_test artifact has invalid shape: {c_test.shape}")

    seq_len = int(model_config["seq_len"])
    forecast_horizon = int(scalers.get("forecast_horizon", seq_len))
    capped_horizon = min(max(int(horizon), 1), seq_len, forecast_horizon)
    capped_path_count = min(max(int(path_count), 1), 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmokeDenoiser(
        input_dim=int(model_config["input_dim"]),
        condition_dim=int(model_config["condition_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        time_dim=int(model_config["time_dim"]),
    ).to(device)
    state_dict = torch.load(paths.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    scheduler = SmokeScheduler(
        num_steps=int(model_config["num_diffusion_steps"]),
        device=device,
    )

    condition_window = c_test[-1]
    condition_batch = np.repeat(condition_window[None, :, :], capped_path_count, axis=0)
    condition_tensor = torch.tensor(condition_batch, dtype=torch.float32, device=device)
    sample_scaled = scheduler.sample_reverse(
        model,
        condition_tensor,
        shape=(capped_path_count, seq_len, int(model_config["input_dim"])),
    ).detach().cpu().numpy()

    return_mean = np.array(scalers["return_scaler_mean"], dtype=np.float64)
    return_scale = np.array(scalers["return_scaler_scale"], dtype=np.float64)
    generated_returns = sample_scaled * return_scale.reshape(1, 1, -1) + return_mean.reshape(1, 1, -1)
    generated_returns = generated_returns[:, :capped_horizon, :]

    condition_mean = np.array(scalers["condition_scaler_mean"], dtype=np.float64)
    condition_scale = np.array(scalers["condition_scaler_scale"], dtype=np.float64)
    latest_condition = condition_window[-1].astype(np.float64) * condition_scale + condition_mean
    start_price = float(latest_condition[0])
    price_paths = start_price * np.exp(np.cumsum(generated_returns, axis=1))

    if not np.isfinite(generated_returns).all() or not np.isfinite(price_paths).all():
        raise RuntimeError("Smoke DDPM generated non-finite output.")

    response = ScenarioGenerationResponse(
        assets=["SPY"],
        paths=generated_returns.tolist(),
        return_scaler_mean=[0.0],
        return_scaler_scale=[1.0],
        start_prices=[start_price],
        horizon=capped_horizon,
        path_count=capped_path_count,
        model_horizon=seq_len,
        live_state=build_live_state_meta(),
        metadata={
            "generator_type": "smoke_ddpm",
            "ddpm_enabled": True,
            "fallback_generator_used": False,
            "smoke_mode": True,
            "assets": ["SPY"],
            "ignored_fields": ["inflation", "interest_rate", "rag_context"],
            "artifact_root": str(paths.root),
            "condition_source": str(paths.c_test),
            "price_path_preview_shape": list(price_paths.shape),
        },
        fallback_generator_used=False,
        generator_type="smoke_ddpm",
        ddpm_enabled=True,
    )
    return response.model_dump()
