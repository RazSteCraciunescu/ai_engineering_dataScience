from __future__ import annotations

import json

import joblib
import pytest

from feedback_intelligence.modeling import predict_priority, predict_resolution


@pytest.fixture()
def sample() -> dict[str, object]:
    return {
        "title": "Workspace reloads during editing",
        "message": "Users lose unsaved work and can reproduce the unexpected refresh.",
        "category": "reliability",
        "channel": "web",
        "product": "workspace",
        "language": "en",
        "status": "open",
        "affected_users": 92,
        "attachment_count": 2,
        "priority": "high",
    }


def test_checked_in_models_can_be_loaded(settings) -> None:
    classifier = joblib.load(settings.models_dir / "priority_classifier.joblib")
    regressor = joblib.load(settings.models_dir / "resolution_regressor.joblib")
    assert hasattr(classifier, "predict")
    assert hasattr(regressor, "predict")


def test_priority_prediction_has_normalized_probabilities(sample) -> None:
    result = predict_priority(sample)
    assert result["priority"] in {"low", "medium", "high"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_resolution_prediction_is_non_negative(sample) -> None:
    result = predict_resolution(sample)
    assert result["estimated_resolution_hours"] >= 0


def test_metrics_are_plausible_not_perfect(settings) -> None:
    metrics = json.loads((settings.reports_dir / "model-metrics.json").read_text())
    classifier = metrics["priority_classifier"]
    regressor = metrics["resolution_regressor"]
    assert 0.70 <= classifier["accuracy"] < 0.99
    assert 0.70 <= classifier["macro_f1"] < 0.99
    assert 0 < regressor["mae_hours"] < 20
    assert 0 < regressor["r2"] < 0.95
