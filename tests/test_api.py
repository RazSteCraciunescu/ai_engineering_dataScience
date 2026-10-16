from __future__ import annotations


def test_root_redirects_to_dashboard(client) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/dashboard"


def test_dashboard_is_served(client) -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "From messy feedback" in response.text


def test_health_is_ready(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_feedback_filter(client) -> None:
    response = client.get("/feedback", params={"limit": 5, "category": "reliability", "priority": "high"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 5
    assert all(item["category"] == "reliability" for item in payload["items"])
    assert all(item["priority"] == "high" for item in payload["items"])


def test_priority_endpoint(client) -> None:
    response = client.post(
        "/predict/priority",
        json={
            "title": "Upload fails without warning",
            "message": "The issue affects many users and retrying does not help.",
            "category": "reliability",
            "affected_users": 120,
            "status": "open",
        },
    )
    assert response.status_code == 200
    assert response.json()["priority"] in {"low", "medium", "high"}


def test_resolution_endpoint(client) -> None:
    response = client.post(
        "/predict/resolution",
        json={
            "title": "Saved report view requested",
            "message": "Several analysts would like to reuse the same filters.",
            "category": "feature request",
            "affected_users": 20,
            "priority": "low",
        },
    )
    assert response.status_code == 200
    assert response.json()["estimated_resolution_hours"] >= 0


def test_search_and_rag_endpoints(client) -> None:
    search = client.post("/search", json={"query": "mean median outliers", "limit": 2})
    rag = client.post("/rag", json={"query": "Why is the median robust to outliers?", "limit": 3})
    assert search.status_code == 200
    assert len(search.json()["results"]) == 2
    assert rag.status_code == 200
    assert rag.json()["evidence"]


def test_input_validation_rejects_short_question(client) -> None:
    response = client.post("/search", json={"query": "x", "limit": 2})
    assert response.status_code == 422


def test_metadata_and_vision_reports(client) -> None:
    metadata = client.get("/metadata")
    vision = client.get("/vision")
    assert metadata.status_code == 200
    assert metadata.json()["model_metrics"]["priority_classifier"]
    assert vision.status_code == 200
    assert len(vision.json()) == 10
