from __future__ import annotations

import json

from feedback_intelligence.pipeline import run


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
