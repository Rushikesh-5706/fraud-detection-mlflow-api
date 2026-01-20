import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/raw_transactions.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Run generate_data.py first")

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Fraud Detection Training")

with mlflow.start_run(run_name="RandomForest_v1") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)
    mlflow.log_metric("accuracy", float(accuracy_score(y_test, y_pred)))
    mlflow.log_metric("precision", float(precision_score(y_test, y_pred, zero_division=0)))
    mlflow.log_metric("recall", float(recall_score(y_test, y_pred, zero_division=0)))
    mlflow.log_metric("f1_score", float(f1_score(y_test, y_pred, zero_division=0)))

    local_model_dir = "model_temp"
    if os.path.exists(local_model_dir):
        import shutil
        shutil.rmtree(local_model_dir)
    mlflow.sklearn.save_model(sk_model=model, path=local_model_dir)
    mlflow.log_artifacts(local_model_dir, artifact_path="model")

    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

print(f"Registered model URI: {model_uri}")

client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
if "FraudDetectionModel" not in [m.name for m in client.search_registered_models()]:
    client.create_registered_model(name="FraudDetectionModel")

model_version = client.create_model_version(name="FraudDetectionModel", source=model_uri, run_id=run_id)
print(f"Model registered: name=FraudDetectionModel, version={model_version.version}")
