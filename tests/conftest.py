from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from feedback_intelligence.api import app
from feedback_intelligence.collection import collect_sources
from feedback_intelligence.config import get_settings
from feedback_intelligence.preparation import prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    return get_settings(ROOT)


@pytest.fixture(scope="session")
def raw_frame(settings) -> pd.DataFrame:
    sources = sorted(
        path for path in settings.raw_dir.iterdir() if path.suffix.lower() in {".csv", ".json", ".html"}
    )
    return collect_sources(sources)


@pytest.fixture(scope="session")
def canonical_frame(raw_frame) -> pd.DataFrame:
    frame, _ = prepare(raw_frame)
    return frame


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def isolated_settings(tmp_path: Path):
    for relative in ("data/raw", "data/knowledge", "sql"):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    return get_settings(tmp_path)
