"""Send retrieved evidence to a local Ollama model without using a hosted API."""
from __future__ import annotations

import json
import urllib.request

from feedback_intelligence.embeddings import search


def ask(question: str, model: str = "llama3.2:3b") -> str:
    evidence = search(question, limit=3)
    context = "\n\n".join(f"SOURCE: {item['source']}\n{item['text']}" for item in evidence)
    prompt = (
        "Answer only from the supplied sources. Say when evidence is insufficient. "
        "Cite source filenames in brackets.\n\n"
        f"QUESTION: {question}\n\nSOURCES:\n{context}"
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["response"]


if __name__ == "__main__":
    print(ask("Why should accuracy be paired with precision and recall?"))
