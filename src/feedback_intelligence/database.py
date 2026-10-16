from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from feedback_intelligence.config import Settings, get_settings


def load_database(frame: pd.DataFrame, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    s.database_path.parent.mkdir(parents=True, exist_ok=True)
    if s.database_path.exists():
        s.database_path.unlink()

    with closing(sqlite3.connect(s.database_path)) as connection:
        connection.executescript((s.root / "sql" / "schema.sql").read_text(encoding="utf-8"))
        export = frame.copy()
        for column in ["satisfaction_was_missing", "is_outlier"]:
            export[column] = export[column].astype(int)
        export.to_sql("feedback", connection, if_exists="append", index=False)
        connection.execute(
            "INSERT INTO pipeline_runs(run_name,row_count,status) VALUES (?,?,?)",
            ("canonical-load", len(export), "passed"),
        )
        connection.commit()
    return s.database_path


def query(
    sql: str, params: tuple[object, ...] = (), settings: Settings | None = None
) -> pd.DataFrame:
    s = settings or get_settings()
    with closing(sqlite3.connect(s.database_path)) as connection:
        return pd.read_sql_query(sql, connection, params=params)
