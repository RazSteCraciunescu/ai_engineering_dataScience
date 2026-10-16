from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_required_portfolio_files_exist() -> None:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml",
        "docs/architecture.md",
        "docs/data-card.md",
        "docs/responsible-ai.md",
        "docs/images/dashboard.png",
        "docs/openapi.json",
        "artifacts/pipeline-manifest.json",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_processed_dataset_has_documented_shape() -> None:
    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    assert frame.shape == (720, 26)
    assert frame["record_id"].is_unique


def test_database_and_csv_row_counts_match() -> None:
    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    with closing(sqlite3.connect(ROOT / "artifacts" / "database" / "feedback_intelligence.sqlite")) as connection:
        count = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    assert count == len(frame)


def test_pipeline_manifest_hashes_are_present() -> None:
    manifest = json.loads((ROOT / "artifacts" / "pipeline-manifest.json").read_text())
    assert manifest["rows"] == 720
    assert all(len(value["sha256"]) == 64 for value in manifest["outputs"].values())


def test_five_notebooks_are_executed_without_errors() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells
        assert all(cell.execution_count is not None for cell in code_cells)
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        assert not errors


def test_no_generated_development_directories_are_committed() -> None:
    blocked = {".git", ".venv", "venv", "node_modules"}
    present = {path.name for path in ROOT.rglob("*") if path.is_dir() and path.name in blocked}
    assert not present
