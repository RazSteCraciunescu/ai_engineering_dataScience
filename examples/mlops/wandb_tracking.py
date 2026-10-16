"""Optional Weights & Biases logging; disabled mode works without an account."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import wandb

    os.environ.setdefault("WANDB_MODE", "disabled")
    metrics = json.loads((ROOT / "artifacts" / "reports" / "model-metrics.json").read_text())
    run = wandb.init(project="feedback-intelligence", config={"seed": 42})
    run.log(
        {
            "accuracy": metrics["priority_classifier"]["accuracy"],
            "macro_f1": metrics["priority_classifier"]["macro_f1"],
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
