"""Optional XGBoost baseline for tabular priority classification."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    from xgboost import XGBClassifier

    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    features = frame[["affected_users", "attachment_count", "text_length", "satisfaction"]]
    labels = LabelEncoder().fit_transform(frame["priority"])
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test), digits=3))


if __name__ == "__main__":
    main()
