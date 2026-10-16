"""Compare normal, uniform and right-skewed distributions."""
from __future__ import annotations

import numpy as np

from feedback_intelligence.statistics import summarize


def main() -> None:
    rng = np.random.default_rng(42)
    samples = {
        "normal": rng.normal(0, 1, 3000),
        "uniform": rng.uniform(-1.7, 1.7, 3000),
        "right_skewed": rng.lognormal(0, 0.75, 3000),
    }
    for name, values in samples.items():
        result = summarize(values)
        print(name, {key: round(value, 3) for key, value in result.items()})


if __name__ == "__main__":
    main()
