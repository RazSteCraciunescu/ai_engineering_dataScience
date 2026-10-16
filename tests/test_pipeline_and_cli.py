from __future__ import annotations

import json
import sys

from feedback_intelligence import cli
from feedback_intelligence.pipeline import run


def test_pipeline_rebuilds_all_core_artifacts(isolated_settings) -> None:
    manifest = run(isolated_settings)
    assert manifest["rows"] == 720
    assert manifest["data_quality"]["cleaning"]["duplicates_removed"] == 6
    assert manifest["retrieval"]["documents"] == 12
    assert manifest["image_count"] == 10
    assert (isolated_settings.reports_dir / "dashboard.html").exists()
    assert (isolated_settings.figures_dir / "correlation-matrix.png").exists()
    assert (isolated_settings.models_dir / "priority_classifier.joblib").exists()


def test_cli_search_outputs_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["feedback-lab", "search", "precision and recall"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert payload[0]["rank"] == 1


def test_cli_ask_outputs_grounded_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["feedback-lab", "ask", "What is data leakage?"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "deterministic-local-rag"
    assert payload["evidence"]
