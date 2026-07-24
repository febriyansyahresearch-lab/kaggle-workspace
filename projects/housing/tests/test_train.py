import pytest
from projects.housing.src.train import train_models


def test_train_models_returns_results():
    results = train_models()
    assert "linear" in results
    assert "random_forest" in results
    assert "gradient_boosting" in results


def test_train_models_r2_positive():
    results = train_models()
    for metrics in results.values():
        assert metrics["r2"] > 0


def test_train_models_rmse_finite():
    results = train_models()
    for metrics in results.values():
        assert metrics["rmse"] > 0
