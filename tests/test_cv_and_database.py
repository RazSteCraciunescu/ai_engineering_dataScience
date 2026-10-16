from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd

from feedback_intelligence.cv_quality import analyze_directory, analyze_image
from feedback_intelligence.database import load_database, query


def test_clear_and_blurred_images_have_different_sharpness(settings) -> None:
    clear = analyze_image(settings.images_dir / "inspection-clear.png")
    blurred = analyze_image(settings.images_dir / "inspection-blurred.png")
    assert clear["laplacian_variance"] > blurred["laplacian_variance"]
    assert blurred["blurred"] is True


def test_dark_image_is_flagged(settings) -> None:
    result = analyze_image(settings.images_dir / "inspection-dark.png")
    assert result["underexposed"] is True


def test_image_directory_report_contains_ten_samples(settings) -> None:
    results = analyze_directory(settings)
    assert len(results) == 10
    assert (settings.reports_dir / "image-quality.csv").exists()


def test_checked_in_database_contains_canonical_rows(settings) -> None:
    result = query("SELECT COUNT(*) AS count FROM feedback", settings=settings)
    assert int(result.iloc[0]["count"]) == 720


def test_category_summary_view_is_queryable(settings) -> None:
    result = query("SELECT * FROM category_summary ORDER BY records DESC", settings=settings)
    assert len(result) == 6
    assert result["records"].sum() == 720


def test_database_can_be_built_in_isolation(canonical_frame: pd.DataFrame, isolated_settings) -> None:
    path = load_database(canonical_frame, isolated_settings)
    with closing(sqlite3.connect(path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    assert count == 720
