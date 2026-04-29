from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXECUTED_DIR = ROOT / "executed_notebooks"

NOTEBOOK_SEQUENCE = [
    "release_aware_macro_loader.ipynb",
    "market_state_and_alignment_builder.ipynb",
    "training_prep_for_upgraded_ddpm.ipynb",
    "retrain_upgraded_ddpm.ipynb",
    "evaluate_upgraded_ddpm.ipynb",
    "live_counterfactual_lab_v1.ipynb",
]


def load_notebook_runtime():
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as exc:
        raise RuntimeError(
            "Notebook execution dependencies are missing. Install them with:\n"
            "pip install nbformat nbclient jupyter ipykernel"
        ) from exc
    return nbformat, NotebookClient


def select_sequence(start_at: str | None, stop_after: str | None) -> list[str]:
    sequence = NOTEBOOK_SEQUENCE[:]

    if start_at:
        if start_at not in sequence:
            raise ValueError(f"--start-at notebook not found in sequence: {start_at}")
        sequence = sequence[sequence.index(start_at):]

    if stop_after:
        if stop_after not in sequence:
            raise ValueError(f"--stop-after notebook not found in active sequence: {stop_after}")
        sequence = sequence[: sequence.index(stop_after) + 1]

    return sequence


def execute_notebook(nbformat, NotebookClient, notebook_path: Path, timeout: int | None, kernel_name: str) -> Path:
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()

    output_path = EXECUTED_DIR / notebook_path.name
    nbformat.write(notebook, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the upgraded counterfactual notebook pipeline in order.")
    parser.add_argument("--start-at", help="Notebook filename to start at.", default=None)
    parser.add_argument("--stop-after", help="Notebook filename to stop after.", default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Per-cell timeout in seconds. Default is no timeout.")
    parser.add_argument("--kernel-name", default="python3", help="Jupyter kernel name to use.")
    args = parser.parse_args()

    nbformat, NotebookClient = load_notebook_runtime()
    sequence = select_sequence(args.start_at, args.stop_after)

    print(f"Workspace: {ROOT}")
    print(f"Executed notebook output dir: {EXECUTED_DIR}")
    print("Notebook run order:")
    for name in sequence:
        print(f"- {name}")

    for name in sequence:
        notebook_path = ROOT / name
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        print()
        print(f"Running {name} ...")
        output_path = execute_notebook(nbformat, NotebookClient, notebook_path, args.timeout, args.kernel_name)
        print(f"Completed {name}")
        print(f"Saved executed notebook: {output_path}")

    print()
    print("Pipeline run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
