import pytest
import pandas as pd
import numpy as np
from src.preprocess import clean_data, encode_categorical, split_data, scale_features


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    result = clean_data(df)
    assert len(result) == 2


def test_clean_data_fills_numeric_nan():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    result = clean_data(df)
    assert result["x"].isna().sum() == 0


def test_clean_data_fills_object_nan():
    df = pd.DataFrame({"y": ["a", None, "b"]})
    result = clean_data(df)
    assert result["y"].isna().sum() == 0


def test_encode_categorical():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    result = encode_categorical(df)
    assert result["color"].dtype in [np.int32, np.int64]


def test_split_data():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
    X_train, X_test, y_train, y_test = split_data(df, target_col="b", test_size=0.25, random_state=42)
    assert len(X_train) == 3
    assert len(X_test) == 1


def test_scale_features():
    import numpy as np
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_test = np.array([[5.0, 6.0]])
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
