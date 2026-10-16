from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from feedback_intelligence.collection import collect_csv, collect_html, collect_json, collect_sources


def test_multisource_collection_has_expected_row_count(raw_frame: pd.DataFrame) -> None:
    assert len(raw_frame) == 726
    assert raw_frame["source_type"].value_counts().to_dict() == {"json": 243, "html": 243, "csv": 240}


def test_collection_adds_provenance(raw_frame: pd.DataFrame) -> None:
    assert raw_frame["source_file"].notna().all()
    assert raw_frame["collected_at"].str.endswith("+00:00").all()


def test_individual_collectors(settings) -> None:
    assert len(collect_csv(settings.raw_dir / "public_feedback.csv")) == 240
    assert len(collect_json(settings.raw_dir / "support_forum.json")) == 243
    assert len(collect_html(settings.raw_dir / "community_board.html")) == 243


def test_unsupported_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "data.xml"
    source.write_text("<rows />", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported source"):
        collect_sources([source])


def test_empty_source_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="No source"):
        collect_sources([])
