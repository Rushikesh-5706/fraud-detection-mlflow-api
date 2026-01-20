import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib

DATA_PATH = Path("data/raw_transactions.csv")
MODEL_OUTPUT_PATH = Path("data/fraud_model.pkl")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run generate_data.py first")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Fraud Detection Training")

    with mlflow.start_run(run_name="RandomForest_v1"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(
            model,
            artifact_path="fraud_model",
            registered_model_name="FraudDetectionModel"
        )

        MODEL_OUTPUT_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)

    print("Model trained locally, logged to MLflow, and fraud_model.pkl created.")
