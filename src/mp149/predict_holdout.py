from pathlib import Path
import joblib
import pandas as pd
FROM_SKL = "sklearn.metrics"
from sklearn.metrics import mean_absolute_error

MODEL_PATH = Path("models/model_v2.joblib")
HOLDOUT_PATH = Path("data/processed/holdout_clean.csv")
OUTPUT_CSV = Path("predictions/holdout_predictions.csv")
OUTPUT_REPORT = Path("predictions/holdout_prediction_report.md")

FEATURES = ["trip_hours", "distance_from_shore_km", "crew", "wave_height_m", "water_temp_c", "net_length_m", "boat", "harbour"]
TARGET_COL = "catch_kg"

def generate_holdout_predictions():
    model = joblib.load(MODEL_PATH)
    holdout_df = pd.read_csv(HOLDOUT_PATH)
    missing = [c for c in FEATURES if c not in holdout_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    preds = model.predict(holdout_df[FEATURES])
    out = pd.DataFrame()
    if "record_id" in holdout_df.columns:
        out["record_id"] = holdout_df["record_id"]
    if "date" in holdout_df.columns:
        out["date"] = holdout_df["date"]
    out["predicted_catch_kg"] = preds.round(4)
    out["model_used"] = "model_v2.joblib"
    has_labels = TARGET_COL in holdout_df.columns and holdout_df[TARGET_COL].notna().all()
    mae_msg = f"Computed MAE: {mean_absolute_error(holdout_df[TARGET_COL], preds):.4f} kg" if has_labels else "Actual catch_kg labels unavailable; holdout predictions generated without calculating MAE."
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    report = f"# Stage 4 Holdout Prediction Report (MP149 - G081)\n\n- Model Used: model_v2.joblib (ExtraTreesRegressor pipeline, check-set MAE 63.5533 kg)\n- Input Dataset: data/processed/holdout_clean.csv ({len(out)} records)\n- Output CSV: predictions/holdout_predictions.csv\n- MAE Status: {mae_msg}\n- Predicted Catch Mean: {out['predicted_catch_kg'].mean():.4f} kg\n- Predicted Catch Min / Max: {out['predicted_catch_kg'].min():.4f} kg / {out['predicted_catch_kg'].max():.4f} kg\n"
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(mae_msg)
    print(f"Saved {len(out)} predictions to {OUTPUT_CSV}")
    return out

if __name__ == "__main__":
    generate_holdout_predictions()
