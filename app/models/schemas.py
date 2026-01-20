from pydantic import BaseModel, Field

class FraudPredictionRequest(BaseModel):
    transaction_amount: float = Field(..., gt=0)
    time_of_day_hour: int = Field(..., ge=0, le=23)
    num_transactions_1h: int = Field(..., ge=0)
    avg_transaction_7d: float = Field(..., gt=0)
    location_risk_score: float = Field(..., ge=0, le=1)

class FraudPredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
