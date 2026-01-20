import logging
from pythonjsonlogger import jsonlogger
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Fraud Detection API"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "FraudDetectionModel"
    model_version: str = "latest"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
