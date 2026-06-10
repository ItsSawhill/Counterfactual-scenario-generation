from pathlib import Path

import pytest

import run_counterfactual_pipeline as runner


def test_smoke_argument_parsing():
    parser = runner.build_parser()

    args = parser.parse_args(["--smoke"])

    assert args.smoke is True
    assert args.start_at is None
    assert args.stop_after is None


def test_smoke_artifact_path_expectations(tmp_path: Path):
    paths = runner.smoke_artifact_paths(tmp_path / "smoke")

    assert paths["root"] == tmp_path / "smoke"
    assert paths["scalers"] == tmp_path / "smoke" / "processed" / "windows" / "scalers_smoke.json"
    assert paths["best_checkpoint"] == tmp_path / "smoke" / "outputs" / "checkpoints" / "conditional_ddpm_smoke_best.pt"
    assert paths["sample_unscaled"] == tmp_path / "smoke" / "outputs" / "generated_samples" / "ddpm_smoke_generated_unscaled.npy"


def test_default_notebook_sequence_remains_unchanged():
    assert runner.NOTEBOOK_SEQUENCE == [
        "release_aware_macro_loader.ipynb",
        "market_state_and_alignment_builder.ipynb",
        "training_prep_for_upgraded_ddpm.ipynb",
        "retrain_upgraded_ddpm.ipynb",
        "evaluate_upgraded_ddpm.ipynb",
        "live_counterfactual_lab_v1.ipynb",
    ]


def test_smoke_cannot_be_combined_with_partial_notebook_run(monkeypatch):
    monkeypatch.setattr(runner, "_run_smoke_pipeline", lambda: 0)
    monkeypatch.setattr("sys.argv", ["run_counterfactual_pipeline.py", "--smoke", "--start-at", "training_prep_for_upgraded_ddpm.ipynb"])

    with pytest.raises(ValueError):
        runner.main()
