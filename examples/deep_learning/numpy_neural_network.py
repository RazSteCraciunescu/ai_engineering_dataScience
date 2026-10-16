"""A tiny two-layer neural network with explicit forward and backward passes."""
from __future__ import annotations

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def train(seed: int = 42, epochs: int = 2500, learning_rate: float = 0.8) -> tuple[np.ndarray, np.ndarray, list[float]]:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.7, size=(2, 5))
    w2 = rng.normal(0, 0.7, size=(5, 1))
    losses: list[float] = []

    for epoch in range(epochs):
        hidden = np.tanh(x @ w1)
        output = sigmoid(hidden @ w2)
        error = output - y
        loss = float(np.mean(error**2))
        if epoch % 100 == 0 or epoch == epochs - 1:
            losses.append(loss)

        output_delta = 2 * error * output * (1 - output) / len(x)
        hidden_delta = (output_delta @ w2.T) * (1 - hidden**2)
        w2 -= learning_rate * hidden.T @ output_delta
        w1 -= learning_rate * x.T @ hidden_delta

    return w1, w2, losses


def main() -> None:
    w1, w2, losses = train()
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    predictions = sigmoid(np.tanh(x @ w1) @ w2).ravel()
    print("predictions:", np.round(predictions, 3).tolist())
    print("loss:", round(losses[-1], 5))


if __name__ == "__main__":
    main()
