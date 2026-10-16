"""Optional SHAP inspection for the trained random-forest resolution model."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import shap

    model = joblib.load(ROOT / "artifacts" / "models" / "resolution_regressor.joblib")
    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv").head(30)
    transformed = model.named_steps["features"].transform(frame[model.feature_names_in_])
    explainer = shap.TreeExplainer(model.named_steps["regressor"])
    values = explainer.shap_values(transformed)
    print("SHAP output shape:", values.shape)


if __name__ == "__main__":
    main()
