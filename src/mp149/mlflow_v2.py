
import hashlib
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient


def main():
    # Load project configuration
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    training = config["training"]
    features = (
        training["numeric_features"]
        + training["categorical_features"]
    )
    target = training["target_column"]

    # Load the original training-data hash
    data_hash = hashlib.sha256(
        Path("data/raw/train.csv").read_bytes()
    ).hexdigest()

    # Load the saved V1 and V2 models
    v1 = joblib.load("models/model.joblib")
    v2 = joblib.load("models/model_v2.joblib")

    # Recreate the same final 30% check set used by train_v2.py
    later = pd.read_csv("data/processed/later_clean.csv")
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date").reset_index(drop=True)

    check = later.iloc[210:]
    X_check = check[features]
    y_check = check[target]

    # Evaluate both models on exactly the same records
    v1_predictions = v1.predict(X_check)
    v2_predictions = v2.predict(X_check)

    v1_mae = mean_absolute_error(y_check, v1_predictions)
    v2_mae = mean_absolute_error(y_check, v2_predictions)

    # Client promotion rule: V2 must improve MAE by at least 10%
    promote_v2 = v2_mae <= 0.90 * v1_mae

    print("V1 check MAE:", v1_mae)
    print("V2 check MAE:", v2_mae)
    print("V2 qualifies for promotion:", promote_v2)

    # Connect to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP149_Fishing_Boat_Catch")

    model_name = "MP149_Fishing_Boat_Catch_Model"

    # Log the V2 evaluation and model
    with mlflow.start_run(run_name="V2_Final_Check") as run:
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("model", "ExtraTreesRegressor")
        mlflow.log_param("training_records", 1210)
        mlflow.log_param("check_records", len(check))
        mlflow.log_param("promotion_rule", "V2_MAE <= 0.90 * V1_MAE")

        mlflow.log_metric("mae_v1_check", v1_mae)
        mlflow.log_metric("mae_v2_check", v2_mae)

        model_info = mlflow.sklearn.log_model(
            v2,
            name="model",
            serialization_format="cloudpickle"
        )

        print("V2 MLflow run ID:", run.info.run_id)

    # Register V2 as a new model version
    model_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name
    )

    client = MlflowClient()

    # Apply the client's promotion decision
    champion_version = (
        model_version.version if promote_v2 else "1"
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=champion_version
    )

    print("Registered V2 model version:", model_version.version)
    print("Champion model version:", champion_version)


if __name__ == "__main__":
    main()
