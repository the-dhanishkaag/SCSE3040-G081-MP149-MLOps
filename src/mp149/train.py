import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import hashlib
import joblib
from mlflow.tracking import MlflowClient

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
def load_config():
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    return config
def get_data_hash():
    file_path = Path("data/raw/train.csv")

    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

    return file_hash
def prepare_data(config):
    data_path = Path(config["data"]["processed_dir"]) / "train_clean.csv"

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    training = config["training"]

    features = (
        training["numeric_features"]
        + training["categorical_features"]
    )

    target = training["target_column"]
    validation_date = training["validation"]["start_date"]

    train_df = df[df["date"] < validation_date]
    val_df = df[df["date"] >= validation_date]

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    return X_train, X_val, y_train, y_val
def create_preprocessor(config):
    training = config["training"]

    numeric_features = training["numeric_features"]
    categorical_features = training["categorical_features"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor
def create_model(experiment):
    model_name = experiment["model"]

    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=experiment["n_estimators"],
            max_depth=experiment["max_depth"],
            random_state=42
        )

    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=experiment["n_estimators"],
            max_depth=experiment["max_depth"],
            random_state=42
        )

    else:
        raise ValueError("Unknown model: " + model_name)

    return model
def run_experiment(experiment, X_train, X_val, y_train, y_val, config, data_hash):

    preprocessor = create_preprocessor(config)
    model = create_model(experiment)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    with mlflow.start_run(run_name=experiment["name"]):

        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("model", experiment["model"])
        mlflow.log_param("n_estimators", experiment["n_estimators"])
        mlflow.log_param("max_depth", experiment["max_depth"])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)

        mlflow.log_metric("mae_validation", mae)

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

        run_id = mlflow.active_run().info.run_id

    print(experiment["name"], "MAE:", mae)

    return pipeline, mae, run_id, model_info.model_uri
def train_models():
    config = load_config()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP149_Fishing_Boat_Catch")

    X_train, X_val, y_train, y_val = prepare_data(config)

    data_hash = get_data_hash()

    results = []

    for experiment in config["training"]["experiments"]:
        pipeline, mae, run_id, model_uri = run_experiment(
            experiment,
            X_train,
            X_val,
            y_train,
            y_val,
            config,
            data_hash
        )

        results.append({
            "name": experiment["name"],
            "pipeline": pipeline,
            "mae": mae,
            "run_id": run_id,
            "model_uri": model_uri
        })

    best_model = min(results, key=lambda result: result["mae"])

    print("\nBest model:", best_model["name"])
    print("Best validation MAE:", best_model["mae"])
    print("Best run ID:", best_model["run_id"])
    registered_model_name = "MP149_Fishing_Boat_Catch_Model"

    model_version = mlflow.register_model(
        model_uri=best_model["model_uri"],
        name=registered_model_name
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=model_version.version
    )

    Path("models").mkdir(exist_ok=True)

    joblib.dump(best_model["pipeline"], "models/model.joblib")

    print("Registered model version:", model_version.version)
    print("Champion alias assigned successfully")

    return best_model
if __name__ == "__main__":
    train_models()