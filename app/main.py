import logging
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api.v1.endpoints import router
from app.core.config import settings, setup_logging

# Initialize logging configuration
setup_logging()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        return json.dumps(log_record)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [handler]

app = FastAPI(title=settings.app_name)

app.include_router(router, prefix="/api/v1")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logging.info(
        f"{request.method} {request.url.path} {response.status_code}"
    )
    return response

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
