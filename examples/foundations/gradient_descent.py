"""Fit y = wx + b with gradient descent instead of a library estimator."""
from __future__ import annotations

import numpy as np

from feedback_intelligence.statistics import gradient_descent


def main() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(-2, 2, 100)
    y = 2.7 * x + 1.4 + rng.normal(0, 0.22, size=x.size)
    result = gradient_descent(x, y, learning_rate=0.04, iterations=1800)
    print(f"weight={result['weight']:.3f}, bias={result['bias']:.3f}")
    print("final sampled loss:", result["loss_history"][-1])


if __name__ == "__main__":
    main()
