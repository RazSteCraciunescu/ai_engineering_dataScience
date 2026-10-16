from __future__ import annotations

from fastapi.testclient import TestClient

from feedback_intelligence.api import app

client = TestClient(app)

health = client.get("/health")
health.raise_for_status()
assert health.json()["status"] == "ok"

feedback = client.get("/feedback", params={"limit": 3, "priority": "high"})
feedback.raise_for_status()
assert len(feedback.json()["items"]) == 3

payload = {
    "title": "Workspace reloads during editing",
    "message": "Several users lose unsaved work when the editor refreshes unexpectedly.",
    "category": "reliability",
    "channel": "web",
    "product": "workspace",
    "affected_users": 87,
    "attachment_count": 2,
    "priority": "high",
}
for endpoint in ("/predict/priority", "/predict/resolution"):
    response = client.post(endpoint, json=payload)
    response.raise_for_status()

search = client.post("/search", json={"query": "precision versus accuracy", "limit": 2})
search.raise_for_status()
assert search.json()["results"]

rag = client.post("/rag", json={"query": "Why can accuracy be misleading?", "limit": 3})
rag.raise_for_status()
assert rag.json()["evidence"]

print("API smoke test passed.")
