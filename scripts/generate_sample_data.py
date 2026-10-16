from __future__ import annotations

import html
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "images"
SEED = 20260314

CATEGORIES = {
    "billing": {
        "titles": ["Invoice total looks wrong", "Refund has not appeared", "Duplicate charge shown"],
        "phrases": ["charged twice", "refund is still pending", "invoice tax does not match"],
    },
    "account": {
        "titles": ["Sign-in loop after reset", "Profile settings will not save", "MFA code arrives late"],
        "phrases": ["cannot complete sign in", "account preferences reset", "verification code expired"],
    },
    "performance": {
        "titles": ["Dashboard is slow at peak time", "Export takes several minutes", "Search results stall"],
        "phrases": ["page remains responsive but slow", "request sometimes times out", "delay is worse in the afternoon"],
    },
    "usability": {
        "titles": ["Filter labels are unclear", "Mobile menu is hard to reach", "Form loses the selected tab"],
        "phrases": ["workflow needs too many clicks", "label does not explain the action", "focus jumps unexpectedly"],
    },
    "feature request": {
        "titles": ["Add bulk export options", "Allow saved report views", "Support a compact table mode"],
        "phrases": ["would reduce repetitive work", "requested by several teammates", "not blocking current usage"],
    },
    "reliability": {
        "titles": ["Upload fails without an error", "Notification job stopped", "Workspace reloads during editing"],
        "phrases": ["issue can be reproduced", "retry sometimes works", "users may lose unsaved changes"],
    },
}

PRODUCTS = ["workspace", "analytics", "mobile", "billing portal"]
CHANNELS = ["web", "mobile", "community"]
STATUSES = ["resolved", "resolved", "resolved", "open", "investigating"]
CATEGORY_ALIASES = {
    "billing": ["billing", "bill"],
    "account": ["account", "login"],
    "performance": ["performance", "slow"],
    "usability": ["usability", "ux"],
    "feature request": ["feature", "feature request"],
    "reliability": ["reliability", "crash"],
}
PRIORITY_ALIASES = {
    "low": ["low", "p3", "minor"],
    "medium": ["medium", "p2", "normal"],
    "high": ["high", "p1", "urgent"],
}
CHANNEL_ALIASES = {"web": ["web", "website"], "mobile": ["mobile", "app"], "community": ["community", "forum"]}


def _priority(rng: np.random.Generator, category: str, users: int, status: str) -> str:
    category_risk = {
        "feature request": 0.0,
        "usability": 0.25,
        "billing": 0.55,
        "performance": 0.95,
        "account": 1.25,
        "reliability": 1.55,
    }[category]
    unresolved_risk = 0.45 if status in {"open", "investigating"} else 0.0
    score = category_risk + np.log1p(users) / 2.0 + unresolved_risk + rng.normal(0, 0.28)
    if score >= 3.65:
        return "high"
    if score >= 2.45:
        return "medium"
    return "low"


def make_records() -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    start = datetime(2025, 1, 3, 8, tzinfo=UTC)
    records: list[dict[str, object]] = []
    category_names = list(CATEGORIES)

    for index in range(720):
        category = category_names[index % len(category_names)]
        spec = CATEGORIES[category]
        channel = str(rng.choice(CHANNELS, p=[0.56, 0.25, 0.19]))
        product = str(rng.choice(PRODUCTS, p=[0.42, 0.24, 0.20, 0.14]))
        status = str(rng.choice(STATUSES))
        users = int(max(1, round(rng.lognormal(3.2, 0.86))))
        if category == "reliability":
            users += int(rng.integers(12, 80))
        attachment_count = int(rng.integers(0, 5))
        priority = _priority(rng, category, users, status)
        satisfaction = float(np.clip(np.rint(rng.normal(3.9 - 0.55 * (priority == "high"), 0.9)), 1, 5))
        if index % 31 == 0:
            satisfaction = np.nan

        complexity = {
            "billing": 8.0,
            "account": 6.0,
            "performance": 14.0,
            "usability": 4.0,
            "feature request": 19.0,
            "reliability": 9.0,
        }[category]
        priority_base = {"low": 32.0, "medium": 19.0, "high": 9.0}[priority]
        status_cost = 8.0 if status == "open" else 12.0 if status == "investigating" else 0.0
        expected_hours = (
            priority_base
            + complexity
            + status_cost
            + 1.6 * np.log1p(users)
            + 1.7 * attachment_count
        )
        if status in {"open", "investigating"} and index % 4 == 0:
            resolution: float | None = None
        else:
            resolution = round(float(max(0.35, expected_hours + rng.normal(0, 6.0))), 2)
        if index in {47, 98, 143, 221, 289, 356, 408, 477, 542, 601, 659, 706}:
            resolution = round(float(rng.uniform(165, 350)), 2)

        title = spec["titles"][index % len(spec["titles"])]
        phrase = spec["phrases"][int(rng.integers(0, len(spec["phrases"])))]
        message = (
            f"A user reports that the {product} {phrase}. "
            f"The report came from the {channel} channel and affects roughly {users} users. "
            f"Reproduction notes were captured for follow-up batch {index % 17 + 1}."
        )
        created = start + timedelta(hours=int(index * 7 + rng.integers(0, 5)))
        record = {
            "source_id": f"FB-{index + 10001}",
            "created_at": created.isoformat(),
            "channel": str(rng.choice(CHANNEL_ALIASES[channel])),
            "product": product.title() if index % 14 == 0 else product,
            "category": str(rng.choice(CATEGORY_ALIASES[category])),
            "title": title,
            "message": message,
            "priority": str(rng.choice(PRIORITY_ALIASES[priority])),
            "satisfaction": satisfaction,
            "resolution_hours": resolution,
            "status": status.title() if index % 12 == 0 else status,
            "language": "EN" if index % 21 == 0 else "en",
            "affected_users": users,
            "attachment_count": attachment_count,
        }
        records.append(record)

    return records


def write_exports(records: list[dict[str, object]]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    # Six records intentionally overlap across exports to exercise canonical deduplication.
    csv_records = records[:240]
    json_records = records[237:480]
    html_records = records[477:]

    pd.DataFrame(csv_records).to_csv(RAW / "public_feedback.csv", index=False)
    json_safe_records = [
        {key: (None if pd.isna(value) else value) for key, value in record.items()}
        for record in json_records
    ]
    (RAW / "support_forum.json").write_text(
        json.dumps(
            {"export": {"generated_by": "local-sample-generator", "records": json_safe_records}},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    headers = list(html_records[0])
    body = []
    for record in html_records:
        cells = "".join(
            f"<td>{html.escape('' if pd.isna(record[key]) else str(record[key]))}</td>" for key in headers
        )
        body.append(f"<tr>{cells}</tr>")
    table = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Community export</title></head><body>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(key)}</th>" for key in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>"
    )
    (RAW / "community_board.html").write_text(table, encoding="utf-8")


def write_images() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1)
    width, height = 640, 360
    base = np.full((height, width, 3), 238, dtype=np.uint8)
    cv2.rectangle(base, (45, 45), (595, 315), (40, 82, 115), 4)
    cv2.circle(base, (170, 180), 74, (68, 176, 155), -1)
    cv2.rectangle(base, (285, 105), (545, 145), (92, 117, 205), -1)
    cv2.rectangle(base, (285, 175), (505, 205), (120, 139, 165), -1)
    cv2.putText(base, "FEEDBACK REVIEW", (275, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 48, 62), 2, cv2.LINE_AA)

    variants = {
        "inspection-clear.png": base,
        "inspection-bright.png": cv2.convertScaleAbs(base, alpha=1.05, beta=40),
        "inspection-dark.png": cv2.convertScaleAbs(base, alpha=0.48, beta=-15),
        "inspection-blurred.png": cv2.GaussianBlur(base, (25, 25), 0),
        "inspection-low-contrast.png": cv2.convertScaleAbs(base, alpha=0.35, beta=120),
    }
    for level in range(5):
        noise = rng.normal(0, 7 + level * 5, base.shape).astype(np.int16)
        variants[f"inspection-noise-{level + 1}.png"] = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for name, image in variants.items():
        if not cv2.imwrite(str(IMAGES / name), image):
            raise RuntimeError(f"Could not write {name}")


def main() -> None:
    records = make_records()
    write_exports(records)
    write_images()
    print(f"Generated {len(records)} unique records, 726 exported rows and 10 inspection images.")


if __name__ == "__main__":
    main()
