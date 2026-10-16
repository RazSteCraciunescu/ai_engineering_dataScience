from __future__ import annotations

import pytest

from feedback_intelligence.embeddings import build_index, load_documents, load_index, search
from feedback_intelligence.rag import answer


def test_knowledge_base_contains_multiple_topics(settings) -> None:
    documents = load_documents(settings.knowledge_dir)
    assert len(documents) == 12
    assert all(document["text"] for document in documents)


def test_index_has_dense_vectors(settings) -> None:
    index = load_index(settings)
    assert index["vectors"].shape[0] == 12
    assert index["vectors"].shape[1] >= 2


def test_search_returns_ranked_evidence(settings) -> None:
    results = search("precision recall accuracy imbalanced classes", limit=3, settings=settings)
    assert [item["rank"] for item in results] == [1, 2, 3]
    assert results[0]["cosine_similarity"] >= results[1]["cosine_similarity"]
    assert results[0]["source"].endswith(".md")


def test_search_exposes_cosine_and_euclidean_metrics(settings) -> None:
    result = search("vector distance embeddings", limit=1, settings=settings)[0]
    assert -1 <= result["cosine_similarity"] <= 1
    assert result["euclidean_distance"] >= 0


def test_blank_search_is_rejected() -> None:
    with pytest.raises(ValueError, match="blank"):
        search("   ")


def test_deterministic_rag_returns_sources() -> None:
    result = answer("Why can accuracy be misleading for an imbalanced dataset?", limit=3)
    assert result["mode"] == "deterministic-local-rag"
    assert result["evidence"]
    assert all("source" in item for item in result["evidence"])


def test_index_can_be_rebuilt_in_isolated_workspace(isolated_settings) -> None:
    metadata = build_index(isolated_settings)
    assert metadata["documents"] == 12
    assert isolated_settings.semantic_index_path.exists()
