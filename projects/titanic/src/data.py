import numpy as np
import pandas as pd


def generate_titanic(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.50])
    age = np.where(pclass == 1, rng.normal(38, 12, n), np.where(pclass == 2, rng.normal(30, 10, n), rng.normal(25, 8, n)))
    age = np.clip(age, 1, 80).astype(int)
    sex = rng.choice(["male", "female"], size=n)
    fare = np.where(pclass == 1, rng.exponential(100, n) + 20, np.where(pclass == 2, rng.exponential(30, n) + 10, rng.exponential(10, n) + 5))
    fare = np.round(fare, 2)
    sibsp = rng.poisson(0.5, size=n)
    parch = rng.poisson(0.3, size=n)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])

    survived = np.where(
        (pclass == 1) & (sex == "female"), rng.binomial(1, 0.95, n),
        np.where(
            (pclass == 2) & (sex == "female"), rng.binomial(1, 0.85, n),
            np.where(
                (pclass == 1) & (sex == "male"), rng.binomial(1, 0.35, n),
                rng.binomial(1, 0.15, n)
            )
        )
    )

    df = pd.DataFrame({
        "pclass": pclass, "age": age, "sex": sex, "fare": fare,
        "sibsp": sibsp, "parch": parch, "embarked": embarked,
        "survived": survived,
    })
    df.loc[rng.choice(n, size=int(n * 0.1)), "age"] = np.nan
    return df
