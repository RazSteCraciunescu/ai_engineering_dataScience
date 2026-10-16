from __future__ import annotations

import html
import json
from string import Template
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from feedback_intelligence.config import Settings, get_settings
from feedback_intelligence.statistics import summarize


def _save(path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def build_reports(frame: pd.DataFrame, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    s.ensure()
    sns.set_theme(context="notebook", style="whitegrid")

    plt.figure(figsize=(8, 4.8))
    order = frame["category"].value_counts().index
    sns.countplot(data=frame, y="category", order=order, color=sns.color_palette()[0])
    plt.title("Feedback volume by category")
    plt.xlabel("records")
    plt.ylabel("")
    _save(s.figures_dir / "category-volume.png")

    plt.figure(figsize=(8, 4.8))
    resolution = frame["resolution_hours"].dropna()
    sns.histplot(resolution, bins=30, kde=True, color=sns.color_palette()[0])
    plt.axvline(resolution.median(), linestyle="--", label="median")
    plt.axvline(resolution.mean(), linestyle=":", label="mean")
    plt.title("Resolution-time distribution")
    plt.xlabel("hours")
    plt.legend()
    _save(s.figures_dir / "resolution-distribution.png")

    plt.figure(figsize=(7.5, 4.8))
    priority_mix = pd.crosstab(frame["category"], frame["priority"], normalize="index")
    sns.heatmap(priority_mix, annot=True, fmt=".2f", cbar=False)
    plt.title("Priority mix within category")
    _save(s.figures_dir / "category-priority-heatmap.png")

    plt.figure(figsize=(7.5, 5.2))
    numeric = [
        "satisfaction",
        "resolution_hours",
        "affected_users",
        "attachment_count",
        "text_length",
    ]
    correlation = frame[numeric].corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, fmt=".2f", center=0)
    plt.title("Numeric correlation matrix")
    _save(s.figures_dir / "correlation-matrix.png")

    summary = {
        "rows": len(frame),
        "categories": frame["category"].value_counts().astype(int).to_dict(),
        "priorities": frame["priority"].value_counts().astype(int).to_dict(),
        "channels": frame["channel"].value_counts().astype(int).to_dict(),
        "resolution_hours": summarize(resolution),
        "satisfaction": summarize(frame["satisfaction"].dropna()),
        "outliers": int(frame["is_outlier"].sum()),
        "missing": {key: int(value) for key, value in frame.isna().sum().items() if value},
    }
    (s.reports_dir / "eda-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_dashboard(
    frame: pd.DataFrame, metrics: dict[str, Any], settings: Settings | None = None
) -> None:
    s = settings or get_settings()
    classifier = metrics["priority_classifier"]
    regressor = metrics["resolution_regressor"]
    categories = frame["category"].value_counts().to_dict()
    maximum = max(categories.values())

    bars = "".join(
        "<div class='bar'><span>{}</span><i><b style='width:{:.1f}%'></b></i>"
        "<strong>{}</strong></div>".format(
            html.escape(str(name)), count / maximum * 100, int(count)
        )
        for name, count in categories.items()
    )
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td><em class='{}'>{}</em></td>"
        "<td>{}</td><td>{}</td></tr>".format(
            html.escape(str(row.record_id)),
            html.escape(str(row.category)),
            html.escape(str(row.priority)),
            html.escape(str(row.priority)),
            html.escape(str(row.channel)),
            html.escape(str(row.title)),
        )
        for row in frame.tail(8).iloc[::-1].itertuples()
    )

    template = Template(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feedback Intelligence</title><style>
:root{--bg:#081018;--panel:#101b27;--line:#223346;--text:#edf5fb;--muted:#91a4b7;--accent:#55d6b3;--a2:#8da8ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}
header{height:70px;display:flex;align-items:center;padding:0 4%;border-bottom:1px solid var(--line);background:#09131d}
header strong{font-size:15px}header small{display:block;color:var(--muted)}nav{margin-left:auto;color:var(--muted);font-size:12px}
main{width:min(1180px,92%);margin:auto}.hero{padding:65px 0 42px}.eyebrow{color:var(--accent);font-size:10px;letter-spacing:.14em;font-weight:800}
h1{font-size:64px;line-height:1.02;letter-spacing:-.04em;margin:10px 0 20px}.intro{color:var(--muted);max-width:720px;font-size:16px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}.card,.panel{background:linear-gradient(145deg,var(--panel),#0d1823);border:1px solid var(--line);border-radius:15px;padding:20px}
.card span,.card small{display:block;color:var(--muted)}.card strong{font-size:28px;display:block;margin:5px 0}.two{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;margin-bottom:16px}
h2{margin:2px 0 18px;font-size:20px}.bar{display:grid;grid-template-columns:110px 1fr 40px;gap:10px;align-items:center;margin:13px 0;font-size:12px}.bar span{color:var(--muted)}
.bar i{height:9px;background:#07101a;border-radius:9px;overflow:hidden}.bar b{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--a2));border-radius:9px}
.score{width:135px;height:135px;border-radius:50%;margin:5px auto 20px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:18px solid var(--accent)}
.score strong{font-size:28px}.score small{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.metrics div{border:1px solid var(--line);border-radius:8px;padding:8px}
.metrics span{display:block;color:var(--muted);font-size:9px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:11px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted)}
em{font-style:normal;padding:2px 8px;border-radius:12px;background:#1b2b3a}em.high{color:#ffd16b}em.medium{color:#9db9ff}em.low{color:var(--accent)}
.concepts{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 70px}.concepts article{border:1px solid var(--line);border-radius:12px;padding:16px;background:#0b151f}
.concepts span{font-size:9px;color:var(--accent)}.concepts strong{display:block;margin:7px 0;font-size:14px}.concepts p{color:var(--muted);font-size:11px;margin:0}
footer{border-top:1px solid var(--line);padding:25px 4%;color:var(--muted);font-size:11px;display:flex;justify-content:space-between}
@media(max-width:800px){h1{font-size:44px}.grid{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.concepts{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><div><strong>Feedback Intelligence</strong><small>Data science + AI engineering</small></div><nav>Overview &nbsp;&nbsp; Models &nbsp;&nbsp; RAG &nbsp;&nbsp; API &nbsp;&nbsp; local mode</nav></header>
<main><section class="hero"><div class="eyebrow">PUBLIC-SAFE MULTISOURCE DATASET</div><h1>From messy feedback<br>to explainable decisions.</h1><p class="intro">A reproducible workspace for collection, EDA, statistics, ML, NLP, computer vision, vector search and local LLM workflows.</p></section>
<section class="grid"><article class="card"><span>Canonical rows</span><strong>$row_count</strong><small>CSV + JSON + HTML</small></article><article class="card"><span>High-priority share</span><strong>$high_share</strong><small>classification target</small></article><article class="card"><span>Median resolution</span><strong>$median_resolution h</strong><small>mean: $mean_resolution h</small></article><article class="card"><span>Detected outliers</span><strong>$outliers</strong><small>IQR rule</small></article></section>
<section class="two"><article class="panel"><div class="eyebrow">EDA SNAPSHOT</div><h2>Category volume</h2>$bars</article><article class="panel"><div class="eyebrow">MODEL QUALITY</div><h2>Held-out evaluation</h2><div class="score"><strong>$f1</strong><small>macro-F1</small></div><div class="metrics"><div><span>Accuracy</span><strong>$accuracy</strong></div><div><span>Precision</span><strong>$precision</strong></div><div><span>Recall</span><strong>$recall</strong></div><div><span>R²</span><strong>$r2</strong></div></div></article></section>
<section class="panel"><div class="eyebrow">RECENT SAMPLE</div><h2>Canonical feedback records</h2><table><thead><tr><th>ID</th><th>Category</th><th>Priority</th><th>Channel</th><th>Title</th></tr></thead><tbody>$rows</tbody></table></section>
<h2 style="margin-top:55px">What the repository demonstrates</h2><section class="concepts"><article><span>MATH</span><strong>Scalars, vectors and tensors</strong><p>Embeddings, masks and numerical structures.</p></article><article><span>STATISTICS</span><strong>Mean, median and distributions</strong><p>Skew, variance, density and outliers.</p></article><article><span>ML</span><strong>Gradient descent and evaluation</strong><p>Leakage boundaries and honest metrics.</p></article><article><span>NLP</span><strong>BERT, ALBERT and RoBERTa</strong><p>Transformer encoders and embeddings.</p></article><article><span>GENERATIVE AI</span><strong>GPT, RAG and Ollama</strong><p>Grounded local-model patterns.</p></article><article><span>COMPUTER VISION</span><strong>OpenCV and Stable Diffusion</strong><p>Classical analysis and local generation.</p></article><article><span>AI ENGINEERING</span><strong>Vector DBs, MCP and tracing</strong><p>Tools, retrieval and observability.</p></article><article><span>DELIVERY</span><strong>FastAPI, Docker and CI/CD</strong><p>Tested artifacts and serving.</p></article></section></main>
<footer><span>Public Feedback Intelligence Platform</span><span>Offline-first reference implementation</span></footer></body></html>"""
    )

    content = template.substitute(
        row_count=len(frame),
        high_share=f"{frame['priority'].eq('high').mean():.1%}",
        median_resolution=f"{frame['resolution_hours'].median():.1f}",
        mean_resolution=f"{frame['resolution_hours'].mean():.1f}",
        outliers=int(frame["is_outlier"].sum()),
        bars=bars,
        rows=rows,
        f1=f"{classifier['macro_f1']:.3f}",
        accuracy=f"{classifier['accuracy']:.3f}",
        precision=f"{classifier['macro_precision']:.3f}",
        recall=f"{classifier['macro_recall']:.3f}",
        r2=f"{regressor['r2']:.3f}",
    )
    (s.reports_dir / "dashboard.html").write_text(content, encoding="utf-8")
