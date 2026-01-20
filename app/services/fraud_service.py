import mlflow
import mlflow.pyfunc
import pandas as pd
import logging
from app.core.config import settings

logger = logging.getLogger("fraud_service")

class FraudDetectionService:
    def __init__(self):
        self.model = None

    def _load_model(self):
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            model_uri = f"models:/{settings.model_name}/{settings.model_version}"
            logger.info(f"Loading model from {model_uri}")
            self.model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.exception("Failed to load MLflow model")
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, payload: dict):
        if self.model is None:
            self._load_model()

        df = pd.DataFrame([payload])

        try:
            prediction = self.model.predict(df)

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(df)
                prob = float(proba[0][1])
            else:
                prob = float(prediction[0])

            is_fraud = bool(prediction[0] == 1)
            return is_fraud, prob

        except Exception as e:
            logger.exception("Prediction failed")
            raise RuntimeError(f"Inference error: {e}")
