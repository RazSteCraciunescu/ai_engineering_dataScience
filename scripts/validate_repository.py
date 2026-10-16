from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import joblib
import nbformat
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "release-validation.json"


def check(condition: bool, message: str, checks: list[dict[str, object]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_files(checks: list[dict[str, object]]) -> None:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "azure-pipelines.yml",
        "docs/architecture.md",
        "docs/data-card.md",
        "docs/data-collection-and-ethics.md",
        "docs/ml-evaluation.md",
        "docs/responsible-ai.md",
        "docs/openapi.json",
        "docs/images/dashboard.png",
        "data/processed/feedback.csv",
        "artifacts/database/feedback_intelligence.sqlite",
        "artifacts/models/priority_classifier.joblib",
        "artifacts/models/resolution_regressor.joblib",
        "artifacts/models/semantic_index.joblib",
        "artifacts/reports/dashboard.html",
        "artifacts/reports/model-metrics.json",
        "artifacts/pipeline-manifest.json",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    check(not missing, f"required files exist; missing={missing}", checks)

    blocked = {".git", ".venv", "venv", "node_modules"}
    found = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_dir() and path.name in blocked)
    check(not found, f"no repository-local environments or Git history; found={found}", checks)


def validate_data(checks: list[dict[str, object]]) -> None:
    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    check(frame.shape == (720, 26), f"canonical dataset shape is 720x26; actual={frame.shape}", checks)
    check(frame["record_id"].is_unique, "record IDs are unique", checks)
    check(set(frame["priority"]) == {"low", "medium", "high"}, "priority domain is normalized", checks)
    check(frame["satisfaction"].between(1, 5).all(), "satisfaction stays within 1..5", checks)
    check(int(frame["is_outlier"].sum()) == 12, "12 extreme durations are flagged and retained", checks)
    check(int(frame["satisfaction_was_missing"].sum()) > 0, "missing-value path is represented", checks)

    raw_counts = {
        "csv": len(pd.read_csv(ROOT / "data" / "raw" / "public_feedback.csv")),
        "json": len(json.loads((ROOT / "data" / "raw" / "support_forum.json").read_text())["export"]["records"]),
        "html": len(pd.read_html(ROOT / "data" / "raw" / "community_board.html")[0]),
    }
    check(raw_counts == {"csv": 240, "json": 243, "html": 243}, f"raw source counts match design; actual={raw_counts}", checks)

    database = ROOT / "artifacts" / "database" / "feedback_intelligence.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
        view_rows = int(connection.execute("SELECT COUNT(*) FROM category_summary").fetchone()[0])
    check(count == 720, f"SQLite row count matches canonical data; actual={count}", checks)
    check(view_rows == 6, f"category summary view has six rows; actual={view_rows}", checks)


def validate_models(checks: list[dict[str, object]]) -> None:
    metrics = json.loads((ROOT / "artifacts" / "reports" / "model-metrics.json").read_text())
    classifier = metrics["priority_classifier"]
    regressor = metrics["resolution_regressor"]
    check(0.70 <= classifier["accuracy"] < 0.99, f"classifier accuracy is plausible; actual={classifier['accuracy']:.4f}", checks)
    check(0.70 <= classifier["macro_f1"] < 0.99, f"classifier macro-F1 is plausible; actual={classifier['macro_f1']:.4f}", checks)
    check(0 < regressor["mae_hours"] < 20, f"regression MAE is bounded; actual={regressor['mae_hours']:.4f}", checks)
    check(0 < regressor["r2"] < 0.95, f"regression R-squared is plausible; actual={regressor['r2']:.4f}", checks)

    classifier_model = joblib.load(ROOT / "artifacts" / "models" / "priority_classifier.joblib")
    regressor_model = joblib.load(ROOT / "artifacts" / "models" / "resolution_regressor.joblib")
    semantic = joblib.load(ROOT / "artifacts" / "models" / "semantic_index.joblib")
    check(hasattr(classifier_model, "predict_proba"), "classifier artifact is loadable", checks)
    check(hasattr(regressor_model, "predict"), "regressor artifact is loadable", checks)
    check(len(semantic["documents"]) == 12, "semantic index contains 12 knowledge documents", checks)
    check(semantic["vectors"].shape[0] == 12, "embedding matrix matches document count", checks)


def validate_notebooks_and_docs(checks: list[dict[str, object]]) -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    check(len(notebooks) == 5, f"five learning notebooks are included; actual={len(notebooks)}", checks)
    for notebook_path in notebooks:
        notebook = nbformat.read(notebook_path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        check(bool(code_cells), f"{notebook_path.name} has executable cells", checks)
        check(all(cell.execution_count is not None for cell in code_cells), f"{notebook_path.name} is executed", checks)
        check(not errors, f"{notebook_path.name} has no stored error output", checks)

    image_path = ROOT / "docs" / "images" / "dashboard.png"
    with Image.open(image_path) as image:
        dimensions = image.size
    check(dimensions[0] >= 1200 and dimensions[1] >= 1600, f"dashboard preview is high resolution; actual={dimensions}", checks)

    openapi = json.loads((ROOT / "docs" / "openapi.json").read_text())
    expected_paths = {"/health", "/feedback", "/predict/priority", "/predict/resolution", "/search", "/rag", "/vision"}
    check(expected_paths <= set(openapi["paths"]), "OpenAPI document includes the core endpoints", checks)


def validate_public_safety(checks: list[dict[str, object]]) -> None:
    text_suffixes = {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".txt", ".sql", ".html", ".example"}
    patterns = {
        "machine-local path": re.compile(r"(?:/mnt/data|[A-Za-z]:\\\\Users\\\\)", re.IGNORECASE),
        "private key block": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "GitHub classic token": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        "AI attribution marker": re.compile(r"(?:generated by (?:chatgpt|gpt)|source[ =:_-]*gpt|as an ai language model)", re.IGNORECASE),
    }
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in patterns.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {label}")
    check(not findings, f"no credentials, machine paths or attribution markers; findings={findings}", checks)


def main() -> None:
    checks: list[dict[str, object]] = []
    try:
        validate_files(checks)
        validate_data(checks)
        validate_models(checks)
        validate_notebooks_and_docs(checks)
        validate_public_safety(checks)
        status = "passed"
        error = None
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    report = {
        "status": status,
        "error": error,
        "checks_passed": sum(bool(item["passed"]) for item in checks),
        "checks_total": len(checks),
        "repository_files": len(files),
        "repository_bytes": sum(path.stat().st_size for path in files),
        "tracked_artifact_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                ROOT / "data" / "processed" / "feedback.csv",
                ROOT / "artifacts" / "database" / "feedback_intelligence.sqlite",
                ROOT / "artifacts" / "models" / "priority_classifier.joblib",
                ROOT / "artifacts" / "models" / "resolution_regressor.joblib",
                ROOT / "artifacts" / "models" / "semantic_index.joblib",
                ROOT / "docs" / "images" / "dashboard.png",
            ]
            if path.exists()
        },
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "error", "checks_passed", "checks_total", "repository_files")}, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
