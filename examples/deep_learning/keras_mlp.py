"""Optional Keras/TensorFlow classifier using the canonical feedback dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import keras
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    numeric = frame[["affected_users", "attachment_count", "text_length"]]
    x = StandardScaler().fit_transform(numeric)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(frame[["priority"]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x.shape[1],)),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(y.shape[1], activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print({"loss": float(loss), "accuracy": float(accuracy), "classes": encoder.categories_[0].tolist()})


if __name__ == "__main__":
    main()
