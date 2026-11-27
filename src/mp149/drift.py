
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m",
]

LOW_DRIFT = 0.10
MODERATE_DRIFT = 0.25
EPSILON = 0.0001
DEFAULT_BINS = 10


def calculate_psi(reference, current, bins=DEFAULT_BINS):
    """
    Calculate Population Stability Index between two distributions.
    Reference data determines the bin boundaries.
    """

    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)

    if reference.empty or current.empty:
        raise ValueError("Both distributions must contain valid values.")

    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Data must not contain infinite values.")

    if bins < 2:
        raise ValueError("Number of bins must be at least 2.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = pd.cut(
        reference, bins=edges, include_lowest=True
    ).value_counts(sort=False).to_numpy()

    current_counts = pd.cut(
        current, bins=edges, include_lowest=True
    ).value_counts(sort=False).to_numpy()

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    reference_pct = np.clip(reference_pct, EPSILON, None)
    current_pct = np.clip(current_pct, EPSILON, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def classify_drift(psi):
    """Classify drift using practical PSI thresholds."""

    if psi < LOW_DRIFT:
        return "low"
    elif psi < MODERATE_DRIFT:
        return "moderate"
    else:
        return "high"


def generate_drift_report(
    reference_path,
    current_path,
    output_path,
    bins=DEFAULT_BINS,
):
    """Compare training and later datasets and save a JSON report."""

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    missing_reference = [
        feature for feature in FEATURES
        if feature not in reference.columns
    ]
    missing_current = [
        feature for feature in FEATURES
        if feature not in current.columns
    ]

    if missing_reference or missing_current:
        raise ValueError(
            f"Missing features. Training: {missing_reference}; "
            f"Later: {missing_current}"
        )

    results = {}

    for feature in FEATURES:
        psi = calculate_psi(
            reference[feature],
            current[feature],
            bins=bins,
        )

        results[feature] = {
            "psi": round(psi, 6),
            "drift_level": classify_drift(psi),
        }

    report = {
        "reference_dataset": str(reference_path),
        "comparison_dataset": str(current_path),
        "reference_rows": len(reference),
        "comparison_rows": len(current),
        "features_analyzed": FEATURES,
        "bins": bins,
        "thresholds": {
            "low": "PSI < 0.10",
            "moderate": "0.10 <= PSI < 0.25",
            "high": "PSI >= 0.25",
        },
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report


if __name__ == "__main__":
    report = generate_drift_report(
        "data/processed/train_clean.csv",
        "data/processed/later_clean.csv",
        "reports/drift_report.json",
    )

    print("Drift analysis completed successfully.")
    print(json.dumps(report["results"], indent=4))
