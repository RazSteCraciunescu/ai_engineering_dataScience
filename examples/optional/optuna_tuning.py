"""Optional Optuna search over a small logistic-regression hyperparameter space."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import optuna

    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    x = frame[["affected_users", "attachment_count", "text_length", "satisfaction"]]
    y = frame["priority"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    def objective(trial: optuna.Trial) -> float:
        model = LogisticRegression(C=trial.suggest_float("C", 0.01, 20, log=True), max_iter=1000)
        model.fit(x_train, y_train)
        return float(f1_score(y_test, model.predict(x_test), average="macro"))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
    print(study.best_value, study.best_params)


if __name__ == "__main__":
    main()
