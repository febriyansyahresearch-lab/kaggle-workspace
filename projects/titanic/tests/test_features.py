import pytest
import pandas as pd
from projects.titanic.src.features import engineer_features


def test_engineer_features_columns():
    df = pd.DataFrame({
        "pclass": [1, 2, 3, 1, 2], "age": [25.0, 30.0, 22.0, 40.0, 35.0],
        "sex": ["male", "female", "male", "female", "male"],
        "fare": [50.0, 20.0, 10.0, 100.0, 30.0],
        "sibsp": [0, 1, 0, 1, 0], "parch": [0, 0, 1, 0, 0],
        "embarked": ["S", "C", "Q", "S", "C"], "survived": [0, 1, 0, 1, 0]
    })
    result = engineer_features(df)
    assert "family_size" in result.columns
    assert "is_alone" in result.columns
    assert "age_bin" in result.columns


def test_engineer_features_no_nan():
    df = pd.DataFrame({
        "pclass": [1, 2, 3], "age": [25.0, None, 30.0],
        "sex": ["male", "female", "male"],
        "fare": [50.0, 20.0, 10.0],
        "sibsp": [0, 1, 0], "parch": [0, 0, 1],
        "embarked": ["S", "C", "Q"], "survived": [0, 1, 0]
    })
    result = engineer_features(df)
    assert result["age"].isna().sum() == 0
