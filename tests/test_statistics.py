from __future__ import annotations

import numpy as np
import pytest

from feedback_intelligence.statistics import (
    binary_metrics,
    euclidean,
    gradient_descent,
    minmax,
    standardize,
    summarize,
)


def test_minmax_maps_range_to_zero_and_one() -> None:
    result = minmax([10, 20, 30])
    assert np.allclose(result, [0, 0.5, 1])


def test_minmax_constant_series_returns_zeros() -> None:
    assert np.allclose(minmax([4, 4, 4]), [0, 0, 0])


def test_standardize_has_zero_mean() -> None:
    result = standardize([2, 4, 8, 10])
    assert np.isclose(result.mean(), 0)
    assert np.isclose(result.std(), 1)


def test_euclidean_distance() -> None:
    assert euclidean([0, 0], [3, 4]) == 5


def test_euclidean_requires_matching_shapes() -> None:
    with pytest.raises(ValueError, match="shapes"):
        euclidean([1, 2], [1])


def test_summary_distinguishes_mean_and_median() -> None:
    result = summarize([1, 2, 3, 100])
    assert result["mean"] > result["median"]
    assert result["skewness"] > 0


def test_gradient_descent_recovers_linear_relationship() -> None:
    x = np.linspace(-2, 2, 80)
    result = gradient_descent(x, 2.5 * x + 1.2, learning_rate=0.04, iterations=1200)
    assert result["weight"] == pytest.approx(2.5, abs=0.03)
    assert result["bias"] == pytest.approx(1.2, abs=0.03)
    assert result["loss_history"][-1] < result["loss_history"][0]


def test_binary_metrics_exposes_precision_recall_tradeoff() -> None:
    result = binary_metrics([1, 1, 1, 0, 0], [1, 0, 1, 1, 0])
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["accuracy"] == pytest.approx(0.6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
