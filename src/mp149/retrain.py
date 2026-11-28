import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error


NUMERIC_FEATURES = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m",
]

CATEGORICAL_FEATURES = ["boat", "harbour"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "catch_kg"


def load_datasets():
    train = pd.read_csv("data/processed/train_clean.csv")
    later = pd.read_csv("data/processed/later_clean.csv")

    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date").reset_index(drop=True)

    return train, later


def split_data(train, later):
    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    check = later.iloc[split_index:].copy()

    training_data = pd.concat(
        [train, later_train],
        ignore_index=True
    )

    return training_data, check


def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    candidate_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", candidate_model),
        ]
    )


def retrain():
    train, later = load_datasets()
    training_data, check = split_data(train, later)

    X_train = training_data[FEATURES]
    y_train = training_data[TARGET]

    X_check = check[FEATURES]
    y_check = check[TARGET]

    print("Training new candidate model...")
    candidate = create_pipeline()
    candidate.fit(X_train, y_train)

    print("Loading existing V2 model...")
    v2 = joblib.load("models/model_v2.joblib")

    candidate_predictions = candidate.predict(X_check)
    v2_predictions = v2.predict(X_check)

    candidate_mae = mean_absolute_error(
        y_check, candidate_predictions
    )
    v2_mae = mean_absolute_error(
        y_check, v2_predictions
    )

    output_path = Path("models/model_retrained.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, output_path)

    print("\nRETRAINING RESULTS")
    print("------------------")
    print("Original training records:", len(train))
    print("Additional training records:", len(training_data) - len(train))
    print("Total training records:", len(training_data))
    print("Reserved evaluation records:", len(check))
    print("Evaluation date range:", check["date"].min(), "to", check["date"].max())

    print("\nV2 MAE:", round(v2_mae, 4))
    print("New candidate MAE:", round(candidate_mae, 4))

    improvement = v2_mae - candidate_mae
    print("MAE improvement:", round(improvement, 4))

    if candidate_mae < v2_mae:
        print("Result: Candidate performs better than V2.")
    elif candidate_mae > v2_mae:
        print("Result: Candidate performs worse than V2.")
    else:
        print("Result: Candidate and V2 have equal MAE.")

    print("\nCandidate saved to:", output_path)


if __name__ == "__main__":
    retrain()
