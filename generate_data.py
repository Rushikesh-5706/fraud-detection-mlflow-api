import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def generate_synthetic_data(num_samples=1000):
    df = pd.DataFrame({
        "transaction_amount": np.random.uniform(10, 5000, num_samples),
        "time_of_day_hour": np.random.randint(0, 24, num_samples),
        "num_transactions_1h": np.random.randint(0, 10, num_samples),
        "avg_transaction_7d": np.random.uniform(50, 1000, num_samples),
        "location_risk_score": np.random.uniform(0.1, 0.9, num_samples),
        "is_fraud": np.random.randint(0, 2, num_samples),
    })

    df.loc[
        (df["transaction_amount"] > 4000) &
        (df["location_risk_score"] > 0.8),
        "is_fraud"
    ] = 1

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    output_path = DATA_DIR / "raw_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated at {output_path}")
