import pandas as pd

from src.mp149.retrain import (
    split_data,
    create_pipeline,
    FEATURES,
    TARGET,
)


def test_split_data_uses_first_70_percent_for_training():
    train = pd.DataFrame({
        "catch_kg": [100, 200]
    })

    later = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=10),
        "catch_kg": range(10),
    })

    training_data, check = split_data(train, later)

    assert len(training_data) == 9
    assert len(check) == 3


def test_split_data_keeps_chronological_order():
    train = pd.DataFrame({"catch_kg": [100]})

    later = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=10),
        "catch_kg": range(10),
    })

    _, check = split_data(train, later)

    assert check["date"].is_monotonic_increasing
    assert check["date"].min() == pd.Timestamp("2026-07-08")


def test_pipeline_has_expected_preprocessing_and_model():
    pipeline = create_pipeline()

    assert pipeline.named_steps["preprocessor"] is not None
    assert pipeline.named_steps["model"].n_estimators == 200
    assert pipeline.named_steps["model"].max_depth == 12
    assert pipeline.named_steps["model"].random_state == 42


def test_feature_and_target_definitions():
    assert len(FEATURES) == 8
    assert TARGET == "catch_kg"
    assert "catch_kg" not in FEATURES
    assert "record_id" not in FEATURES
