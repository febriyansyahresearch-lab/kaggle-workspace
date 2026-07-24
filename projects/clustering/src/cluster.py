import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from projects.clustering.src.data import generate_customer_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def find_optimal_k(X, max_k: int = 10) -> list[float]:
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias


def run_clustering(n_clusters: int = 3) -> dict:
    df = generate_customer_data()
    features = df[["annual_income", "spending_score", "age", "purchase_frequency"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    return {
        "labels": labels.tolist(),
        "inertia": float(kmeans.inertia_),
        "pca_explained": pca.explained_variance_ratio_.tolist(),
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    result = run_clustering()
    print(f"Clusters: {result['n_clusters']}, Inertia: {result['inertia']:.2f}")
    print(f"PCA explained variance: {result['pca_explained']}")
