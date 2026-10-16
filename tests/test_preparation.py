from __future__ import annotations

import pandas as pd
import pytest

from feedback_intelligence.preparation import prepare, validate


def test_preparation_removes_only_cross_export_duplicates(raw_frame: pd.DataFrame) -> None:
    frame, report = prepare(raw_frame)
    assert report["raw_rows"] == 726
    assert report["canonical_rows"] == 720
    assert report["duplicates_removed"] == 6
    assert frame["record_id"].is_unique


def test_categories_priorities_and_channels_are_normalized(canonical_frame: pd.DataFrame) -> None:
    assert set(canonical_frame["category"]) == {
        "billing",
        "account",
        "performance",
        "usability",
        "feature request",
        "reliability",
    }
    assert set(canonical_frame["priority"]) == {"low", "medium", "high"}
    assert set(canonical_frame["channel"]) == {"web", "mobile", "community"}


def test_missing_satisfaction_is_imputed_and_marked(canonical_frame: pd.DataFrame) -> None:
    missing = canonical_frame[canonical_frame["satisfaction_was_missing"]]
    assert len(missing) > 0
    assert missing["satisfaction_raw"].isna().all()
    assert missing["satisfaction"].between(1, 5).all()


def test_outliers_are_preserved_but_capped_for_training(canonical_frame: pd.DataFrame) -> None:
    outliers = canonical_frame[canonical_frame["is_outlier"]]
    assert len(outliers) == 12
    assert (outliers["resolution_hours_raw"] == outliers["resolution_hours"]).all()
    assert (outliers["resolution_hours_model"] < outliers["resolution_hours"]).all()


def test_validation_passes_for_canonical_data(canonical_frame: pd.DataFrame) -> None:
    result = validate(canonical_frame)
    assert result["ok"] is True
    assert all(result["checks"].values())


def test_missing_required_columns_are_rejected(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        prepare(raw_frame.drop(columns=["message"]))
