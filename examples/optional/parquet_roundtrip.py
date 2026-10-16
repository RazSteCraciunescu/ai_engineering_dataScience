"""Optional Parquet round trip showing a columnar analytical format."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "data" / "processed" / "feedback.csv"
    target = ROOT / "artifacts" / "reports" / "feedback.parquet"
    frame = pd.read_csv(source)
    frame.to_parquet(target, index=False)
    restored = pd.read_parquet(target)
    assert frame.shape == restored.shape
    print(target.relative_to(ROOT), restored.shape)


if __name__ == "__main__":
    main()
