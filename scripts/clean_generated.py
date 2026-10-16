from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for relative in ("artifacts/database", "artifacts/figures", "artifacts/models", "artifacts/reports", "data/processed"):
    target = ROOT / relative
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

print("Generated artifacts cleared. Raw sample inputs were preserved.")
