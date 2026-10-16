"""Optional Hugging Face inference; model download is intentionally opt-in."""
from __future__ import annotations


def main() -> None:
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    samples = [
        "The export is much faster after the update.",
        "The page keeps reloading and I lost my work.",
    ]
    for text, result in zip(samples, classifier(samples), strict=True):
        print(text, result)


if __name__ == "__main__":
    main()
