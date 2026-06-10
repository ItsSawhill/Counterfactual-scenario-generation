from pathlib import Path

import pipeline_preflight_check as preflight


def _configure_temp_preflight(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "counterfactual_data_build"
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(preflight, "REQUIRED_FILES", [])
    monkeypatch.setattr(preflight, "REQUIRED_PACKAGES", {})
    monkeypatch.setattr(
        preflight,
        "DDPM_ARTIFACTS",
        [
            (
                "Processed data directory",
                artifact_root / "processed",
                True,
                "create processed data",
            ),
            (
                "Checkpoint directory",
                artifact_root / "outputs" / "checkpoints",
                True,
                "create checkpoints",
            ),
            (
                "Best DDPM checkpoint",
                artifact_root / "outputs" / "checkpoints" / "conditional_ddpm_upgraded_best.pt",
                True,
                "train checkpoint",
            ),
            (
                "Generated samples directory",
                artifact_root / "outputs" / "generated_samples",
                True,
                "generate samples",
            ),
            (
                "RAG vector store directory",
                tmp_path / ".rag_store",
                False,
                "ingest docs",
            ),
        ],
    )


def test_preflight_default_warns_but_exits_zero_for_missing_artifacts(monkeypatch, tmp_path, capsys):
    _configure_temp_preflight(monkeypatch, tmp_path)

    exit_code = preflight.main(["--skip-network"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[MISSING] Processed data directory" in captured.out
    assert "Fallback FastAPI/React demo mode can still run" in captured.out


def test_preflight_strict_exits_nonzero_for_missing_ddpm_artifacts(monkeypatch, tmp_path, capsys):
    _configure_temp_preflight(monkeypatch, tmp_path)

    exit_code = preflight.main(["--skip-network", "--strict"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Strict preflight failed" in captured.out
    assert "Best DDPM checkpoint" in captured.out
