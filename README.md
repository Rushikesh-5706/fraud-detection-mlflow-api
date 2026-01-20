# Fraud Detection API — Production-Ready ML Inference Service

## Overview
This project implements a production-ready, containerized fraud detection inference API built with FastAPI and MLflow. It demonstrates a complete MLOps workflow: synthetic data generation, model training, experiment tracking, model registration, and scalable model serving using Docker Compose.

The system exposes REST endpoints for health checks and fraud prediction, validates inputs using Pydantic, logs structured application events, and loads a registered model directly from the MLflow Model Registry at runtime. The setup is fully reproducible on any machine with Docker installed.

---

## Key Features
- RESTful API built with FastAPI
- Strict request validation using Pydantic schemas
- Pre-trained scikit-learn fraud detection model
- MLflow experiment tracking and Model Registry integration
- Model loading via `models:/` URI from MLflow
- Structured JSON logging for application events
- Multi-stage Docker build for optimized image size
- Docker Compose orchestration with persistent MLflow backend
- Unit and integration test coverage

---

## Architecture Summary
- **API Layer**: FastAPI application exposing `/api/v1/health` and `/api/v1/predict`
- **Service Layer**: Centralized fraud detection service responsible for model loading and inference
- **ML Lifecycle**: Synthetic data → training → MLflow tracking → model registration → runtime loading
- **Infrastructure**: Dockerized services orchestrated via Docker Compose

---

## Project Structure
```
.
├── app/
│   ├── main.py
│   ├── api/v1/endpoints.py
│   ├── services/fraud_service.py
│   ├── models/schemas.py
│   └── core/
│       ├── config.py
│       └── logging_config.py
├── tests/
│   ├── unit/test_fraud_service.py
│   └── integration/test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── generate_data.py
├── train_model.py
├── .env.example
└── README.md
```

---

## Prerequisites
- Docker
- Docker Compose

No local Python environment setup is required when using Docker.

---

## Environment Configuration
The application uses environment variables for configuration.

An example file is provided:
```
.env.example
```

Key variables:
- `MLFLOW_TRACKING_URI` – MLflow tracking server URL
- `MODEL_NAME` – Registered model name (default: `FraudDetectionModel`)
- `MODEL_VERSION` – Model version to load (default: `1`)

---

## Running the System (End-to-End)

### 1. Build and Start Services
```
docker compose up --build -d
```

This starts:
- Fraud Detection API on **http://localhost:8000**
- MLflow Tracking UI on **http://localhost:5000**

---

### 2. Generate Synthetic Data
```
docker compose exec fraud-api python generate_data.py
```

This creates a synthetic dataset at `data/raw_transactions.csv` inside the container.

---

### 3. Train and Register the Model
```
docker compose exec fraud-api python train_model.py
```

This step:
- Trains a scikit-learn model
- Logs parameters and metrics to MLflow
- Registers the model as `FraudDetectionModel` (version `1`) in the MLflow Model Registry

---

### 4. Verify API Endpoints

#### Health Check
```
curl -i http://localhost:8000/api/v1/health
```
Expected response:
```
HTTP/1.1 200 OK
{"status": "ok"}
```

#### Valid Prediction
```
curl -i -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 4200,
    "time_of_day_hour": 23,
    "num_transactions_1h": 6,
    "avg_transaction_7d": 300,
    "location_risk_score": 0.9
  }'
```

#### Invalid Input (Validation Error)
```
curl -i -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{ "transaction_amount": -10 }'
```
Expected status: `422 Unprocessable Entity`

---

## Screenshots

All screenshots are available in the `screenshots/` directory and document the full execution flow.

| Screenshot | Description |
|----------|-------------|
| `01_docker_compose_ps.png` | Docker services running successfully |
| `02_mlflow_experiment.png` | MLflow experiment with logged metrics |
| `03_mlflow_model_registry.png` | Registered model visible in MLflow Model Registry |
| `04_health_200.png` | Successful `/health` endpoint response |
| `05_predict_200.png` | Valid `/predict` request returning a prediction |
| `06_predict_422.png` | Input validation error (`422 Unprocessable Entity`) |
| `07_mlflow_registry_cli.png` | Registered model verified via MLflow client API |

---

## Docker Image
A prebuilt Docker image is available:

**Docker Hub**: https://hub.docker.com/r/rushi5706/fraud-detection-api

---

## Testing

### Run Tests Locally (inside container)
```
docker compose exec fraud-api pytest
```

Test coverage includes:
- Unit tests for fraud prediction logic
- Integration tests for API endpoints and error handling

---

## Data and Artifacts
The following directories are intentionally excluded from version control:
- `data/`
- `mlruns/`
- `mlflow.db`

They are recreated automatically via the provided scripts.

---

## Video Demonstration

A full walkthrough video demonstrating setup, training, MLflow UI, and API usage can be added here:

```
<VIDEO_DEMO_URL>
```

---

## Conclusion
This project delivers a complete, reproducible fraud detection inference system that follows modern MLOps best practices. It is designed to be easily evaluated, deployed, and extended for real-world use cases.

