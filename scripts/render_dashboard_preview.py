from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "dashboard.png"
WIDTH, HEIGHT = 1440, 1980
BG = "#081018"
PANEL = "#101b27"
LINE = "#223346"
TEXT = "#edf5fb"
MUTED = "#91a4b7"
ACCENT = "#55d6b3"
ACCENT_2 = "#8da8ff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=PANEL, outline=LINE, width=2)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text.upper(), font=font(15, True), fill=ACCENT)


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "processed" / "feedback.csv")
    metrics = json.loads((ROOT / "artifacts" / "reports" / "model-metrics.json").read_text())
    model = metrics["priority_classifier"]
    reg = metrics["resolution_regressor"]

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, WIDTH, 82), fill="#09131d")
    draw.rounded_rectangle((55, 23, 96, 64), radius=10, fill=ACCENT)
    draw.text((116, 20), "Feedback Intelligence", font=font(22, True), fill=TEXT)
    draw.text((116, 49), "Data science + AI engineering", font=font(13), fill=MUTED)
    draw.text((1060, 32), "Overview   Models   RAG   API   Local mode", font=font(14), fill=MUTED)

    label(draw, (60, 128), "Public-safe multisource dataset")
    draw.text((58, 162), "From messy feedback", font=font(62, True), fill=TEXT)
    draw.text((58, 235), "to explainable decisions.", font=font(62, True), fill=TEXT)
    intro = "A reproducible workspace for collection, EDA, statistics, ML, NLP, computer vision, vector search and local LLM workflows."
    draw.text((60, 325), intro, font=font(20), fill=MUTED)

    cards = [
        ("Canonical rows", f"{len(frame):,}", "CSV + JSON + HTML"),
        ("High-priority share", f"{frame['priority'].eq('high').mean():.1%}", "classification target"),
        ("Median resolution", f"{frame['resolution_hours'].median():.1f} h", f"mean: {frame['resolution_hours'].mean():.1f} h"),
        ("Detected outliers", str(int(frame["is_outlier"].sum())), "IQR rule; raw values kept"),
    ]
    x0, y0, gap, card_w, card_h = 60, 385, 18, 315, 142
    for index, (title, value, note) in enumerate(cards):
        x = x0 + index * (card_w + gap)
        rounded(draw, (x, y0, x + card_w, y0 + card_h))
        draw.text((x + 22, y0 + 20), title, font=font(15), fill=MUTED)
        draw.text((x + 22, y0 + 50), value, font=font(32, True), fill=TEXT)
        draw.text((x + 22, y0 + 103), note, font=font(13), fill=MUTED)

    left = (60, 550, 830, 900)
    right = (850, 550, 1380, 900)
    rounded(draw, left)
    rounded(draw, right)
    label(draw, (85, 575), "EDA snapshot")
    draw.text((85, 605), "Category volume", font=font(24, True), fill=TEXT)
    categories = frame["category"].value_counts()
    max_count = int(categories.max())
    for index, (name, count) in enumerate(categories.items()):
        y = 660 + index * 38
        draw.text((85, y), str(name).title(), font=font(14), fill=MUTED)
        draw.rounded_rectangle((260, y + 4, 720, y + 19), radius=7, fill="#07101a")
        fill_w = int(460 * int(count) / max_count)
        draw.rounded_rectangle((260, y + 4, 260 + fill_w, y + 19), radius=7, fill=ACCENT_2)
        draw.text((742, y - 2), str(int(count)), font=font(15, True), fill=TEXT)

    label(draw, (875, 575), "Model quality")
    draw.text((875, 605), "Held-out evaluation", font=font(24, True), fill=TEXT)
    draw.ellipse((1000, 655, 1228, 883), outline=LINE, width=28)
    draw.arc((1000, 655, 1228, 883), -90, -90 + int(360 * model["macro_f1"]), fill=ACCENT, width=28)
    score = f"{model['macro_f1']:.3f}"
    bbox = draw.textbbox((0, 0), score, font=font(35, True))
    draw.text((1114 - (bbox[2] - bbox[0]) / 2, 735), score, font=font(35, True), fill=TEXT)
    draw.text((1072, 784), "macro-F1", font=font(14), fill=MUTED)
    metric_lines = [
        f"Accuracy  {model['accuracy']:.3f}",
        f"Precision  {model['macro_precision']:.3f}",
        f"Recall  {model['macro_recall']:.3f}",
        f"MAE  {reg['mae_hours']:.2f} h",
    ]
    for index, line in enumerate(metric_lines):
        draw.text((880, 675 + index * 43), line, font=font(15, index == 0), fill=TEXT if index == 0 else MUTED)

    table_box = (60, 925, 1380, 1275)
    rounded(draw, table_box)
    label(draw, (85, 950), "Recent sample")
    draw.text((85, 980), "Canonical feedback records", font=font(24, True), fill=TEXT)
    columns = [(85, "ID"), (250, "Category"), (450, "Priority"), (610, "Channel"), (760, "Title")]
    for x, name in columns:
        draw.text((x, 1030), name, font=font(14, True), fill=MUTED)
    draw.line((85, 1060, 1355, 1060), fill=LINE, width=2)
    for row_index, row in enumerate(frame.tail(6).iloc[::-1].itertuples()):
        y = 1075 + row_index * 31
        values = [row.record_id, row.category.title(), row.priority.title(), row.channel.title(), row.title]
        for (x, _), value in zip(columns, values, strict=True):
            draw.text((x, y), str(value)[:62], font=font(13), fill=TEXT if x in {85, 760} else MUTED)
        draw.line((85, y + 25, 1355, y + 25), fill=LINE, width=1)

    draw.text((60, 1325), "What the repository demonstrates", font=font(28, True), fill=TEXT)
    concepts = [
        ("MATH", "Scalars, vectors and tensors", "Embeddings, masks and array operations."),
        ("STATISTICS", "Mean, median and distributions", "Skew, variance, density and outliers."),
        ("MACHINE LEARNING", "Gradient descent and evaluation", "Leakage controls and honest metrics."),
        ("NLP", "BERT, ALBERT and RoBERTa", "Transformer encoders and embeddings."),
        ("GENERATIVE AI", "GPT, RAG and Ollama", "Grounded local-model patterns."),
        ("COMPUTER VISION", "OpenCV and Stable Diffusion", "Classical analysis and local generation."),
        ("AI ENGINEERING", "Vector DBs, MCP and tracing", "Retrieval, tools and observability."),
        ("DELIVERY", "FastAPI, Docker and CI/CD", "Tested artifacts and serving."),
    ]
    concept_w, concept_h = 315, 190
    for index, (kind, title, description) in enumerate(concepts):
        col, row = index % 4, index // 4
        x = 60 + col * (concept_w + 18)
        y = 1380 + row * (concept_h + 18)
        rounded(draw, (x, y, x + concept_w, y + concept_h), radius=14)
        label(draw, (x + 20, y + 20), kind)
        title_lines = wrap(title, width=25)
        for line_index, line in enumerate(title_lines):
            draw.text((x + 20, y + 53 + line_index * 25), line, font=font(18, True), fill=TEXT)
        description_y = y + 115 if len(title_lines) == 2 else y + 92
        for line_index, line in enumerate(wrap(description, width=35)):
            draw.text((x + 20, description_y + line_index * 20), line, font=font(13), fill=MUTED)

    draw.line((0, 1835, WIDTH, 1835), fill=LINE, width=2)
    draw.text((60, 1875), "Public Feedback Intelligence Platform", font=font(14), fill=MUTED)
    draw.text((1090, 1875), "Offline-first reference implementation", font=font(14), fill=MUTED)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT.relative_to(ROOT), image.size, OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
