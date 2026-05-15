from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_rag_query_handles_missing_vector_store_gracefully(monkeypatch):
    from backend import main

    def _raise(*args, **kwargs):
        raise RuntimeError("Vector store not found. Run POST /rag/ingest before querying.")

    monkeypatch.setattr(main, "retrieve_context", _raise)
    response = client.post("/rag/query", json={"query": "inflation shock", "top_k": 3})
    assert response.status_code == 400
    assert "Vector store not found" in response.json()["detail"]
