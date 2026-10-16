"""Compact comparison of transformer families used in NLP and generation."""
from __future__ import annotations

FAMILIES = {
    "BERT": "Bidirectional encoder; masked-language-model pretraining; strong for understanding tasks.",
    "ALBERT": "Parameter-sharing and factorized embeddings reduce BERT-style model size.",
    "RoBERTa": "BERT encoder trained with revised masking and optimization choices.",
    "GPT": "Autoregressive decoder trained to predict the next token; suited to generation.",
}


def main() -> None:
    for name, description in FAMILIES.items():
        print(f"{name}: {description}")


if __name__ == "__main__":
    main()
