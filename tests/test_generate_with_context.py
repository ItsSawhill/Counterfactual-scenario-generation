from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_generate_with_context_returns_scenario_and_context(monkeypatch):
    from backend import main
    from backend.rag.schemas import QueryResponse, RetrievedContext

    monkeypatch.setattr(
        main,
        "retrieve_context",
        lambda *args, **kwargs: QueryResponse(
            query="stagflation",
            contexts=[
                RetrievedContext(
                    text="Inflation reacceleration can pressure equity multiples.",
                    source="docs/results.md",
                    score=0.81,
                )
            ],
        ),
    )

    response = client.post(
        "/generate-with-context",
        json={
            "user_request": "Generate a stagflation scenario.",
            "top_k": 1,
            "scenario_parameters": {
                "inflation": 0.055,
                "interest_rate": 0.0525,
                "horizon": 20,
                "path_count": 32,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_request"] == "Generate a stagflation scenario."
    assert payload["retrieved_context"]
    assert payload["generated_scenario"]["paths"]
    assert payload["fallback_generator_used"] is True
