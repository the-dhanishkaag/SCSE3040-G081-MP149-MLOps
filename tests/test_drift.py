
import json

import numpy as np
import pandas as pd
import pytest

from src.mp149.drift import (
    FEATURES,
    calculate_psi,
    classify_drift,
    generate_drift_report,
)


def test_identical_distributions_have_zero_psi():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    psi = calculate_psi(data, data)

    assert psi == pytest.approx(0.0)


def test_different_distributions_have_positive_psi():
    reference = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    current = pd.Series([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

    psi = calculate_psi(reference, current)

    assert psi > 0


def test_psi_handles_missing_values():
    reference = pd.Series([1, 2, 3, 4, 5, np.nan])
    current = pd.Series([1, 2, 3, 4, 5, np.nan])

    psi = calculate_psi(reference, current)

    assert psi == pytest.approx(0.0)


def test_psi_rejects_infinite_values():
    reference = pd.Series([1, 2, 3, 4, 5])
    current = pd.Series([1, 2, 3, 4, np.inf])

    with pytest.raises(ValueError):
        calculate_psi(reference, current)


def test_psi_rejects_invalid_bin_count():
    reference = pd.Series([1, 2, 3, 4, 5])

    with pytest.raises(ValueError):
        calculate_psi(reference, reference, bins=1)


def test_drift_classification():
    assert classify_drift(0.05) == "low"
    assert classify_drift(0.15) == "moderate"
    assert classify_drift(0.30) == "high"


def test_generate_drift_report(tmp_path):
    reference = pd.DataFrame({
        feature: np.arange(1, 101)
        for feature in FEATURES
    })

    current = reference.copy()

    reference_path = tmp_path / "train.csv"
    current_path = tmp_path / "later.csv"
    output_path = tmp_path / "drift_report.json"

    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    report = generate_drift_report(
        reference_path,
        current_path,
        output_path,
    )

    assert output_path.exists()
    assert report["reference_rows"] == 100
    assert report["comparison_rows"] == 100
    assert len(report["results"]) == 6

    with output_path.open(encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report["results"]["trip_hours"]["psi"] == 0.0
