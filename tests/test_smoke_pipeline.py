from pathlib import Path
import sys
import types

import pytest

import run_counterfactual_pipeline as runner


def test_smoke_argument_parsing():
    parser = runner.build_parser()

    args = parser.parse_args(["--smoke", "--disable-mlflow"])

    assert args.smoke is True
    assert args.disable_mlflow is True
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
    monkeypatch.setattr(runner, "_run_smoke_pipeline", lambda enable_mlflow=True: 0)
    monkeypatch.setattr("sys.argv", ["run_counterfactual_pipeline.py", "--smoke", "--start-at", "training_prep_for_upgraded_ddpm.ipynb"])

    with pytest.raises(ValueError):
        runner.main()


def test_smoke_mlflow_logging_can_be_disabled(monkeypatch):
    captured = {}

    def fake_smoke_pipeline(enable_mlflow=True):
        captured["enable_mlflow"] = enable_mlflow
        return 0

    monkeypatch.setattr(runner, "_run_smoke_pipeline", fake_smoke_pipeline)
    monkeypatch.setattr("sys.argv", ["run_counterfactual_pipeline.py", "--smoke", "--disable-mlflow"])

    assert runner.main() == 0
    assert captured["enable_mlflow"] is False


def test_smoke_mlflow_uses_local_tracking_uri(monkeypatch, tmp_path):
    calls = {"params": None, "metrics": None, "artifacts": []}

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_mlflow = types.SimpleNamespace(
        set_tracking_uri=lambda value: calls.update({"tracking_uri": value}),
        set_experiment=lambda value: calls.update({"experiment": value}),
        start_run=lambda run_name: calls.update({"run_name": run_name}) or FakeRun(),
        log_params=lambda value: calls.update({"params": value}),
        log_metrics=lambda value: calls.update({"metrics": value}),
        log_artifact=lambda path, artifact_path=None: calls["artifacts"].append((path, artifact_path)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(runner, "MLFLOW_TRACKING_DIR", tmp_path / "mlruns")

    paths = runner.smoke_artifact_paths(tmp_path / "smoke")
    for key in ("windows_dir", "tables_dir", "checkpoints_dir", "samples_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["model_config"].write_text("{}", encoding="utf-8")
    paths["scalers"].write_text("{}", encoding="utf-8")
    paths["best_checkpoint"].write_text("checkpoint", encoding="utf-8")
    paths["sample_scaled"].write_bytes(b"scaled")
    paths["sample_unscaled"].write_bytes(b"unscaled")
    paths["training_summary"].write_text("{}", encoding="utf-8")

    tracking_dir = runner._log_smoke_mlflow_run(
        paths=paths,
        model_config={
            "seq_len": 10,
            "input_dim": 1,
            "condition_dim": 3,
            "hidden_dim": 24,
            "num_diffusion_steps": 20,
        },
        scaler_payload={"forecast_horizon": 10},
        summary={
            "num_epochs_completed": 2,
            "final_train_loss": 0.12,
            "final_val_loss": None,
            "total_seconds": 3.4,
            "train_samples": 100,
            "val_samples": 20,
            "test_samples": 20,
        },
    )

    assert tracking_dir == tmp_path / "mlruns"
    assert calls["tracking_uri"] == (tmp_path / "mlruns").as_uri()
    assert calls["experiment"] == runner.MLFLOW_EXPERIMENT_NAME
    assert calls["run_name"] == runner.SMOKE_MLFLOW_RUN_NAME
    assert calls["params"]["generator_mode"] == "smoke_ddpm"
    assert calls["metrics"]["final_train_loss"] == 0.12
    assert "final_val_loss" not in calls["metrics"]
    assert calls["artifacts"]
