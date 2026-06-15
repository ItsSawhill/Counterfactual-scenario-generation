import pytest

from backend.model.generator_registry import (
    FallbackSimulationGenerator,
    SmokeDDPMGenerator,
    get_generator,
    get_generator_runtime_status,
)


def test_registry_selects_fallback(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "fallback")
    monkeypatch.setenv("SMOKE_DDPM_ENABLED", "1")

    generator = get_generator()
    status = get_generator_runtime_status()

    assert isinstance(generator, FallbackSimulationGenerator)
    assert generator.generator_type == "fallback_simulation"
    assert generator.ddpm_enabled is False
    assert status["selected_generator"] == "fallback_simulation"
    assert status["ddpm_enabled"] is False


def test_registry_selects_smoke_ddpm(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "smoke_ddpm")
    monkeypatch.delenv("SMOKE_DDPM_ENABLED", raising=False)

    generator = get_generator()
    status = get_generator_runtime_status()

    assert isinstance(generator, SmokeDDPMGenerator)
    assert generator.generator_type == "smoke_ddpm"
    assert generator.ddpm_enabled is True
    assert status["selected_generator"] == "smoke_ddpm"
    assert "ready" in status["artifact_status"]


def test_registry_preserves_legacy_smoke_env_when_mode_missing(monkeypatch):
    monkeypatch.delenv("GENERATOR_MODE", raising=False)
    monkeypatch.setenv("SMOKE_DDPM_ENABLED", "1")

    generator = get_generator()

    assert isinstance(generator, SmokeDDPMGenerator)
    assert generator.generator_type == "smoke_ddpm"


def test_invalid_generator_mode_produces_clear_error(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "full_ddpm")

    with pytest.raises(RuntimeError, match="Invalid GENERATOR_MODE='full_ddpm'"):
        get_generator()
