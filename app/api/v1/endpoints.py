from fastapi import APIRouter, HTTPException, status
from app.models.schemas import FraudPredictionRequest, FraudPredictionResponse
from app.services.fraud_service import FraudDetectionService

router = APIRouter()

fraud_service = FraudDetectionService()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {"status": "ok"}

@router.post("/predict", response_model=FraudPredictionResponse)
async def predict(request: FraudPredictionRequest):
    try:
        is_fraud, fraud_probability = fraud_service.predict(
            request.model_dump()
        )
        return FraudPredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_probability
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
