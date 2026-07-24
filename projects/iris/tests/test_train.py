import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from projects.iris.src.train import train_model, evaluate_model


def test_train_model_returns_rf():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y, n_estimators=10, random_state=42)
    assert isinstance(model, RandomForestClassifier)


def test_evaluate_model_returns_accuracy():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y, n_estimators=10, random_state=42)
    result = evaluate_model(model, X, y)
    assert "accuracy" in result
    assert "report" in result
    assert result["accuracy"] > 0
