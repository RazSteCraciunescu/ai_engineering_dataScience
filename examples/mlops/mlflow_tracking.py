"""Optional local MLflow run with no remote tracking server."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import mlflow

    metrics = json.loads((ROOT / "artifacts" / "reports" / "model-metrics.json").read_text())
    mlflow.set_tracking_uri((ROOT / "artifacts" / "mlruns").as_uri())
    mlflow.set_experiment("feedback-priority")
    with mlflow.start_run():
        mlflow.log_param("model", metrics["priority_classifier"]["model"])
        mlflow.log_metric("accuracy", metrics["priority_classifier"]["accuracy"])
        mlflow.log_metric("macro_f1", metrics["priority_classifier"]["macro_f1"])
        mlflow.log_artifact(ROOT / "artifacts" / "reports" / "model-metrics.json")


if __name__ == "__main__":
    main()
