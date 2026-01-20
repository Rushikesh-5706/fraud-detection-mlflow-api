import pytest
from app.services.fraud_service import FraudDetectionService

class DummyModel:
    def predict(self, df):
        return [1]

    def predict_proba(self, df):
        return [[0.1, 0.9]]

def test_predict_success(monkeypatch):
    service = FraudDetectionService.__new__(FraudDetectionService)
    service.model = DummyModel()

    payload = {
        "transaction_amount": 1500.0,
        "time_of_day_hour": 14,
        "num_transactions_1h": 2,
        "avg_transaction_7d": 350.0,
        "location_risk_score": 0.6,
    }

    is_fraud, prob = service.predict(payload)

    assert is_fraud is True
    assert prob == 0.9
