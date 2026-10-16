"""Optional LlamaIndex setup using the local knowledge directory."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120)
    documents = SimpleDirectoryReader(str(ROOT / "data" / "knowledge")).load_data()
    engine = VectorStoreIndex.from_documents(documents).as_query_engine(similarity_top_k=3)
    print(engine.query("Why preserve raw outliers during data preparation?"))


if __name__ == "__main__":
    main()
