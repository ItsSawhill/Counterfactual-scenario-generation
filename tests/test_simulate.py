from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_simulate_returns_frontend_compatible_json():
    response = client.post(
        "/simulate",
        json={
            "inflation": 0.03,
            "interest_rate": 0.04,
            "horizon": 15,
            "path_count": 24,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"] == ["SPY", "QQQ", "GC", "CL", "ZN"]
    assert len(payload["paths"]) == 24
    assert len(payload["paths"][0]) == 15
    assert len(payload["paths"][0][0]) == 5
    assert len(payload["return_scaler_mean"]) == 5
    assert len(payload["return_scaler_scale"]) == 5
    assert payload["horizon"] == 15
    assert payload["path_count"] == 24
