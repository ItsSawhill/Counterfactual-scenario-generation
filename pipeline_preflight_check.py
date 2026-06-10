from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import os
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "counterfactual_data_build"

REQUIRED_FILES = [
    ROOT / "dataset_spec_v1.json",
    ROOT / "scenario_definitions_v1.csv",
    ROOT / "ablation_matrix_v1.csv",
    ROOT / "release_aware_macro_loader.ipynb",
    ROOT / "market_state_and_alignment_builder.ipynb",
    ROOT / "training_prep_for_upgraded_ddpm.ipynb",
    ROOT / "retrain_upgraded_ddpm.ipynb",
    ROOT / "evaluate_upgraded_ddpm.ipynb",
    ROOT / "live_counterfactual_lab_v1.ipynb",
]

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "yfinance": "yfinance",
    "requests": "requests",
    "IPython": "ipython",
    "nbformat": "nbformat",
    "nbclient": "nbclient",
}

NETWORK_HOSTS = {
    "FRED": ("fred.stlouisfed.org", 443),
    "Yahoo Finance": ("query1.finance.yahoo.com", 443),
}

DDPM_ARTIFACTS = [
    (
        "Processed data directory",
        ARTIFACT_ROOT / "processed",
        True,
        "Run release_aware_macro_loader.ipynb, market_state_and_alignment_builder.ipynb, and training_prep_for_upgraded_ddpm.ipynb.",
    ),
    (
        "Checkpoint directory",
        ARTIFACT_ROOT / "outputs" / "checkpoints",
        True,
        "Run retrain_upgraded_ddpm.ipynb.",
    ),
    (
        "Best DDPM checkpoint",
        ARTIFACT_ROOT / "outputs" / "checkpoints" / "conditional_ddpm_upgraded_best.pt",
        True,
        "Run retrain_upgraded_ddpm.ipynb to create the trained checkpoint.",
    ),
    (
        "Generated samples directory",
        ARTIFACT_ROOT / "outputs" / "generated_samples",
        True,
        "Run retrain_upgraded_ddpm.ipynb and evaluate_upgraded_ddpm.ipynb.",
    ),
    (
        "RAG vector store directory",
        ROOT / ".rag_store",
        False,
        "Start the backend and run POST /rag/ingest when retrieval endpoints are needed.",
    ),
]


@dataclass(frozen=True)
class ArtifactStatus:
    label: str
    path: Path
    exists: bool
    required_for_ddpm: bool
    hint: str


def check_required_files() -> list[str]:
    issues: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            issues.append(f"Missing file: {path}")
    return issues


def check_packages() -> tuple[list[str], list[str]]:
    missing: list[str] = []
    present: list[str] = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
        else:
            present.append(module_name)
    return present, missing


def check_network(timeout_seconds: float = 3.0) -> list[str]:
    issues: list[str] = []
    for label, (host, port) in NETWORK_HOSTS.items():
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                pass
        except OSError as exc:
            issues.append(f"{label} unreachable ({host}:{port}) -> {exc}")
    return issues


def check_artifacts() -> list[ArtifactStatus]:
    return [
        ArtifactStatus(
            label=label,
            path=path,
            exists=path.exists(),
            required_for_ddpm=required_for_ddpm,
            hint=hint,
        )
        for label, path, required_for_ddpm, hint in DDPM_ARTIFACTS
    ]


def missing_required_ddpm_artifacts(artifact_statuses: list[ArtifactStatus]) -> list[ArtifactStatus]:
    return [
        status
        for status in artifact_statuses
        if status.required_for_ddpm and not status.exists
    ]


def ensure_output_dirs() -> list[Path]:
    created = [
        ARTIFACT_ROOT / "processed" / "macro",
        ARTIFACT_ROOT / "processed" / "market",
        ARTIFACT_ROOT / "processed" / "aligned",
        ARTIFACT_ROOT / "processed" / "windows",
        ARTIFACT_ROOT / "outputs" / "tables",
        ARTIFACT_ROOT / "outputs" / "checkpoints",
        ARTIFACT_ROOT / "outputs" / "generated_samples",
        ARTIFACT_ROOT / "outputs" / "figures",
        ARTIFACT_ROOT / "outputs" / "live_counterfactuals",
        ROOT / "executed_notebooks",
    ]
    for path in created:
        path.mkdir(parents=True, exist_ok=True)
    return created


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for the upgraded counterfactual pipeline.")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network reachability checks for FRED and Yahoo Finance.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when required DDPM artifacts are missing. Default mode warns so the fallback demo can still run.",
    )
    args = parser.parse_args(argv)

    print(f"Workspace: {ROOT}")
    print(f"Python: {sys.executable}")

    print_section("Artifact Readiness")
    artifact_statuses = check_artifacts()
    for status in artifact_statuses:
        relative_path = status.path
        try:
            relative_path = status.path.relative_to(ROOT)
        except ValueError:
            pass
        if status.exists:
            print(f"[PASS] {status.label}: {relative_path}")
        elif status.required_for_ddpm:
            print(f"[MISSING] {status.label}: {relative_path}")
            print(f"         {status.hint}")
        else:
            print(f"[WARN] {status.label}: {relative_path}")
            print(f"       {status.hint}")

    created_dirs = ensure_output_dirs()
    print_section("Output Directories")
    for path in created_dirs:
        print(f"[ok] {path}")

    print_section("Required Files")
    file_issues = check_required_files()
    if file_issues:
        for issue in file_issues:
            print(f"[missing] {issue}")
    else:
        for path in REQUIRED_FILES:
            print(f"[ok] {path.name}")

    print_section("Python Packages")
    _, missing_packages = check_packages()
    if missing_packages:
        print("Missing packages:")
        for package_name in missing_packages:
            print(f"- {package_name}")
        print()
        print("Suggested install command:")
        print(
            "pip install " + " ".join(sorted(set(missing_packages)))
        )
    else:
        print("All required packages found.")

    print_section("Environment")
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key:
        print("[ok] FRED_API_KEY detected. ALFRED / FRED vintage mode can be used where supported.")
    else:
        print("[warn] FRED_API_KEY not set. The macro loader will fall back to release-lag approximation mode.")

    if not args.skip_network:
        print_section("Network Reachability")
        network_issues = check_network()
        if network_issues:
            for issue in network_issues:
                print(f"[warn] {issue}")
        else:
            for label in NETWORK_HOSTS:
                print(f"[ok] {label}")
    else:
        network_issues = []

    has_errors = bool(file_issues or missing_packages)
    strict_artifact_errors = missing_required_ddpm_artifacts(artifact_statuses)
    print_section("Summary")
    if has_errors:
        print("Preflight found blockers. Fix the missing files and packages before running the full pipeline.")
        return 1

    if args.strict and strict_artifact_errors:
        print("Strict preflight failed. Required DDPM artifacts are missing:")
        for status in strict_artifact_errors:
            print(f"- {status.label}: {status.path}")
        return 1

    if strict_artifact_errors:
        print("Preflight passed for source/package setup, but DDPM artifacts are missing.")
        print("Fallback FastAPI/React demo mode can still run. Use --strict to fail on missing DDPM artifacts.")
        return 0

    if network_issues:
        print("Preflight passed with warnings. Package and file setup is fine, but network access may block live data downloads.")
        return 0

    print("Preflight passed. The workspace is ready for a first full run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
