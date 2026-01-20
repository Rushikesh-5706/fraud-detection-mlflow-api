from fastapi import FastAPI
from app.api.v1 import endpoints
from app.core.config import settings
from app.core.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.include_router(endpoints.router, prefix="/api/v1", tags=["Fraud Prediction"])

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup")
    import mlflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("MLflow tracking configured")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")
