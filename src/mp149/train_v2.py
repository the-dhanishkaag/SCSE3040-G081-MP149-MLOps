import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

# Load cleaned datasets
train = pd.read_csv("data/processed/train_clean.csv")
later = pd.read_csv("data/processed/later_clean.csv")

# Sort later data chronologically
later["date"] = pd.to_datetime(later["date"])
later = later.sort_values("date").reset_index(drop=True)

# Split later data: first 70% for training, last 30% for checking
later_train = later.iloc[:210].copy()
check = later.iloc[210:].copy()

# Combine all original training data with newer training data
v2_train = pd.concat([train, later_train], ignore_index=True)

# Define features and target
numeric_features = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m"
]

categorical_features = ["boat", "harbour"]

features = numeric_features + categorical_features
target = "catch_kg"

X_train = v2_train[features]
y_train = v2_train[target]

X_check = check[features]
y_check = check[target]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Create v2 model using the best v1 model configuration
model = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train v2
print("Training v2 model...")
pipeline.fit(X_train, y_train)

# Evaluate v1 and v2 on exactly the same check records
v1 = joblib.load("models/model.joblib")

v1_predictions = v1.predict(X_check)
v2_predictions = pipeline.predict(X_check)

v1_mae = mean_absolute_error(y_check, v1_predictions)
v2_mae = mean_absolute_error(y_check, v2_predictions)

# Save v2 separately
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/model_v2.joblib")

print("\nV2 training completed!")
print("Original training records:", len(train))
print("Additional later training records:", len(later_train))
print("Total v2 training records:", len(v2_train))
print("Final check records:", len(check))
print("Check date range:", check["date"].min(), "to", check["date"].max())

print("\nV1 MAE on check:", round(v1_mae, 4))
print("V2 MAE on check:", round(v2_mae, 4))

print("\nMAE improvement:", round(v1_mae - v2_mae, 4))

if v2_mae < v1_mae:
    print("V2 performs better on the check set.")
else:
    print("V2 does not outperform V1 on the check set.")

print("\nV2 saved to models/model_v2.joblib")
