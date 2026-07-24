import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["age_bin"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 50, 80], labels=[0, 1, 2, 3, 4]).astype(int)
    df["fare_bin"] = pd.qcut(df["fare"] + 1e-6, q=4, labels=[0, 1, 2, 3]).astype(int)
    le = LabelEncoder()
    df["sex"] = le.fit_transform(df["sex"])
    df["embarked"] = le.fit_transform(df["embarked"].astype(str))
    return df
