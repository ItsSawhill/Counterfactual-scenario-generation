from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "vector_store_ready" in payload
