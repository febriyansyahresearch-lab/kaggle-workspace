import numpy as np
import pandas as pd


def generate_customer_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cluster_centers = {
        0: {"age": 25, "income": 35, "spend": 20, "freq": 5},
        1: {"age": 45, "income": 80, "spend": 70, "freq": 15},
        2: {"age": 35, "income": 50, "spend": 40, "freq": 8},
    }

    rows = []
    for _ in range(n):
        c = rng.choice([0, 1, 2])
        center = cluster_centers[c]
        rows.append({
            "age": int(np.clip(rng.normal(center["age"], 5), 18, 70)),
            "annual_income": float(np.clip(rng.normal(center["income"], 10), 15, 150)),
            "spending_score": float(np.clip(rng.normal(center["spend"], 10), 1, 100)),
            "purchase_frequency": int(np.clip(rng.normal(center["freq"], 2), 1, 30)),
        })
    return pd.DataFrame(rows)
