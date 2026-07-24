import pytest
from projects.titanic.src.train import train


def test_train_returns_metrics():
    metrics = train(n_estimators=10)
    assert "accuracy" in metrics
    assert "report" in metrics
    assert metrics["accuracy"] > 0
