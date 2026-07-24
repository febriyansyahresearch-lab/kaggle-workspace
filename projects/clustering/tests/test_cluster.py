import pytest
import numpy as np
from projects.clustering.src.cluster import run_clustering, find_optimal_k
from projects.clustering.src.data import generate_customer_data


def test_run_clustering_returns_labels():
    result = run_clustering(n_clusters=3)
    assert "labels" in result
    assert len(result["labels"]) == 300


def test_run_clustering_inertia_decreases():
    from sklearn.preprocessing import StandardScaler
    df = generate_customer_data(200)
    X = StandardScaler().fit_transform(df)
    inertias = find_optimal_k(X, max_k=5)
    assert len(inertias) == 5
    assert inertias[0] > inertias[-1]
