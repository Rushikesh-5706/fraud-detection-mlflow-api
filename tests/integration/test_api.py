from fastapi.testclient import TestClient
from app.main import app

class DummyModel:
    def predict(self, df):
        return [0]

    def predict_proba(self, df):
        return [[0.8, 0.2]]

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_payload(monkeypatch):
    monkeypatch.setattr(
        "app.api.v1.endpoints.fraud_service.model",
        DummyModel()
    )

    payload = {
        "transaction_amount": 500.0,
        "time_of_day_hour": 9,
        "num_transactions_1h": 1,
        "avg_transaction_7d": 200.0,
        "location_risk_score": 0.2
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is False
    assert body["fraud_probability"] == 0.2

def test_predict_invalid_payload():
    payload = {
        "transaction_amount": -10,
        "time_of_day_hour": 30,
        "num_transactions_1h": -1,
        "avg_transaction_7d": -100,
        "location_risk_score": 2
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
