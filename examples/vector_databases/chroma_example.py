"""Optional persistent Chroma collection populated from local Markdown knowledge files."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(ROOT / "artifacts" / "chroma"))
    collection = client.get_or_create_collection("feedback-knowledge")
    paths = sorted((ROOT / "data" / "knowledge").glob("*.md"))
    collection.upsert(
        ids=[path.stem for path in paths],
        documents=[path.read_text(encoding="utf-8") for path in paths],
        metadatas=[{"source": path.name} for path in paths],
    )
    print(collection.query(query_texts=["precision versus accuracy"], n_results=3))


if __name__ == "__main__":
    main()
