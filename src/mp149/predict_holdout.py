
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "MP149_Fishing_Boat_Catch_Model"
MODEL_ALIAS = "champion"

# File paths
HOLDOUT_PATH = Path("data/processed/holdout_clean.csv")
OUTPUT_CSV = Path("predictions/holdout_predictions.csv")
OUTPUT_REPORT = Path("predictions/holdout_prediction_report.md")

# Model input features
FEATURES = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m",
    "boat",
    "harbour",
]


def generate_holdout_predictions():

    # Load the model assigned the champion alias
    client = MlflowClient()
    champion = client.get_model_version_by_alias(
        MODEL_NAME, MODEL_ALIAS
    )

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load holdout data
    holdout_df = pd.read_csv(HOLDOUT_PATH)

    # Validate required columns
    required_columns = ["record_id"] + FEATURES
    missing = [
        column for column in required_columns
        if column not in holdout_df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if holdout_df["record_id"].isna().any():
        raise ValueError("Holdout contains missing record_id values")

    # Generate predictions
    predictions = model.predict(holdout_df[FEATURES])

    # Create exactly two required output columns
    output = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": predictions.round(4),
    })

    if len(output) != len(holdout_df):
        raise ValueError("Prediction count does not match holdout rows")

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    # Generate report
    report = f"""# Stage 4 Holdout Prediction Report (MP149 - G081)

- Model Used: {MODEL_NAME}@{MODEL_ALIAS}
- MLflow Model Version: {champion.version}
- Input Dataset: {HOLDOUT_PATH} ({len(output)} records)
- Output CSV: {OUTPUT_CSV}
- Prediction Mean: {output['prediction'].mean():.4f} kg
- Prediction Min: {output['prediction'].min():.4f} kg
- Prediction Max: {output['prediction'].max():.4f} kg
- Output Columns: record_id, prediction
"""

    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(f"Champion model version: {champion.version}")
    print(f"Saved {len(output)} predictions to {OUTPUT_CSV}")
    print(f"Output columns: {list(output.columns)}")
    print(f"Prediction mean: {output['prediction'].mean():.4f} kg")

    return output


if __name__ == "__main__":
    generate_holdout_predictions()
