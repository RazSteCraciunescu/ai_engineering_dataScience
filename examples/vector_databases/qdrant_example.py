"""Optional in-memory Qdrant example using deterministic numeric vectors."""
from __future__ import annotations

import numpy as np


def main() -> None:
    from qdrant_client import QdrantClient, models

    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(8, 12)).astype(float)
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="concepts",
        vectors_config=models.VectorParams(size=12, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="concepts",
        points=[
            models.PointStruct(id=index, vector=vector.tolist(), payload={"concept": f"concept-{index}"})
            for index, vector in enumerate(vectors)
        ],
    )
    print(client.query_points(collection_name="concepts", query=vectors[0].tolist(), limit=3).points)


if __name__ == "__main__":
    main()
