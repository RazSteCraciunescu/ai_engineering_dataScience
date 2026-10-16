from __future__ import annotations

import json
from pathlib import Path

from feedback_intelligence.api import app

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "docs" / "openapi.json"
target.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
print(target.relative_to(ROOT))
