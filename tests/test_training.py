import pytest
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from src.mp149.train import (
    create_model,
    create_preprocessor,
    load_config,
    prepare_data,
)
from sklearn.metrics import mean_absolute_error


def test_load_config():
    config = load_config()
    assert config is not None
    assert "training" in config


def test_create_random_forest():
    experiment = {
        "model": "random_forest",
        "n_estimators": 100,
        "max_depth": 5
    }

    model = create_model(experiment)

    assert model is not None
    assert model.n_estimators == 100
    assert model.max_depth == 5


def test_create_extra_trees():
    experiment = {
        "model": "extra_trees",
        "n_estimators": 100,
        "max_depth": 10
    }

    model = create_model(experiment)

    assert model is not None
    assert model.n_estimators == 100
    assert model.max_depth == 10


def test_invalid_model_name():
    experiment = {
        "model": "invalid_model",
        "n_estimators": 100,
        "max_depth": 5
    }

    with pytest.raises(ValueError):
        create_model(experiment)


def test_preprocessor_creation():
    config = load_config()

    preprocessor = create_preprocessor(config)

    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2


def test_training_validation_split():
    config = load_config()

    X_train, X_val, y_train, y_val = prepare_data(config)

    assert not X_train.empty
    assert not X_val.empty
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_mae_calculation():
    actual = [100, 200, 300]
    predicted = [110, 190, 310]

    mae = mean_absolute_error(actual, predicted)

    assert mae == pytest.approx(10.0)
    assert mae >= 0

def test_run_experiment_integration(monkeypatch):
    import src.mp149.train as train_module

    config = load_config()

    X_train, X_val, y_train, y_val = prepare_data(config)

    experiment = {
        "name": "test_integration",
        "model": "random_forest",
        "n_estimators": 5,
        "max_depth": 3
    }

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", lambda *args, **kwargs: type("MockInfo", (), {"model_uri": "mock_uri"})())
    monkeypatch.setattr(train_module.mlflow, "active_run", lambda: type("MockRun", (), {"info": type("MockInfo", (), {"run_id": "test_run"})()})())

    class MockContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(train_module.mlflow, "start_run", lambda **kwargs: MockContext())

    pipeline, mae, run_id, model_uri = train_module.run_experiment(
        experiment,
        X_train,
        X_val,
        y_train,
        y_val,
        config,
        "test_hash"
    )

    assert pipeline is not None
    assert mae >= 0
    assert run_id == "test_run"
    assert model_uri == "mock_uri"