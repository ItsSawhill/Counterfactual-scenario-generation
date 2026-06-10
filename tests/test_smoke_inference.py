from pathlib import Path

import numpy as np
import pytest

from backend.model.smoke_inference import (
    SmokeArtifactPaths,
    SmokeArtifactsMissingError,
    generate_smoke_ddpm_response,
    smoke_artifacts_ready,
)


def test_direct_smoke_inference_loads_checkpoint_and_produces_finite_output():
    if not smoke_artifacts_ready():
        pytest.skip("Smoke artifacts are missing. Run `python run_counterfactual_pipeline.py --smoke`.")

    payload = generate_smoke_ddpm_response(path_count=4, horizon=7)

    assert payload["assets"] == ["SPY"]
    assert payload["generator_type"] == "smoke_ddpm"
    assert payload["ddpm_enabled"] is True
    assert payload["fallback_generator_used"] is False
    assert payload["horizon"] == 7
    assert payload["path_count"] == 4

    paths = np.asarray(payload["paths"], dtype=float)
    assert paths.shape == (4, 7, 1)
    assert np.isfinite(paths).all()
    assert np.isfinite(np.asarray(payload["start_prices"], dtype=float)).all()


def test_smoke_inference_missing_artifacts_raises_clear_error(tmp_path: Path):
    missing_paths = SmokeArtifactPaths(root=tmp_path / "missing_smoke")

    with pytest.raises(SmokeArtifactsMissingError, match="Run `python run_counterfactual_pipeline.py --smoke` first"):
        generate_smoke_ddpm_response(path_count=4, horizon=7, paths=missing_paths)
