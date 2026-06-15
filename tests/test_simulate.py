from fastapi.testclient import TestClient

from backend.main import app
from backend.model.smoke_inference import smoke_artifacts_ready


client = TestClient(app)


def test_simulate_returns_frontend_compatible_json(monkeypatch):
    monkeypatch.delenv("GENERATOR_MODE", raising=False)
    monkeypatch.delenv("SMOKE_DDPM_ENABLED", raising=False)
    response = client.post(
        "/simulate",
        json={
            "inflation": 0.03,
            "interest_rate": 0.04,
            "horizon": 15,
            "path_count": 24,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"] == ["SPY", "QQQ", "GC", "CL", "ZN"]
    assert len(payload["paths"]) == 24
    assert len(payload["paths"][0]) == 15
    assert len(payload["paths"][0][0]) == 5
    assert len(payload["return_scaler_mean"]) == 5
    assert len(payload["return_scaler_scale"]) == 5
    assert payload["horizon"] == 15
    assert payload["path_count"] == 24
    assert payload["generator_type"] == "fallback_simulation"
    assert payload["ddpm_enabled"] is False
    assert payload["fallback_generator_used"] is True


def test_simulate_smoke_ddpm_mode_returns_checkpoint_backed_metadata(monkeypatch):
    if not smoke_artifacts_ready():
        import pytest

        pytest.skip("Smoke artifacts are missing. Run `python run_counterfactual_pipeline.py --smoke`.")

    monkeypatch.delenv("GENERATOR_MODE", raising=False)
    monkeypatch.setenv("SMOKE_DDPM_ENABLED", "1")
    response = client.post(
        "/simulate",
        json={
            "inflation": 0.10,
            "interest_rate": 0.01,
            "horizon": 30,
            "path_count": 200,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"] == ["SPY"]
    assert payload["generator_type"] == "smoke_ddpm"
    assert payload["ddpm_enabled"] is True
    assert payload["fallback_generator_used"] is False
    assert payload["metadata"]["smoke_mode"] is True
    assert payload["metadata"]["ignored_fields"] == ["inflation", "interest_rate", "rag_context"]
    assert payload["horizon"] == 10
    assert payload["path_count"] == 128
    assert len(payload["paths"]) == 128
    assert len(payload["paths"][0]) == 10
    assert len(payload["paths"][0][0]) == 1


def test_simulate_invalid_generator_mode_returns_clear_error(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "full_ddpm")

    response = client.post(
        "/simulate",
        json={
            "inflation": 0.03,
            "interest_rate": 0.04,
            "horizon": 15,
            "path_count": 24,
        },
    )

    assert response.status_code == 503
    assert "Invalid GENERATOR_MODE='full_ddpm'" in response.json()["detail"]
