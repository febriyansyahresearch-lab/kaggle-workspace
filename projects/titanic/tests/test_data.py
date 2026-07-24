import pytest
from projects.titanic.src.data import generate_titanic


def test_generate_titanic_shape():
    df = generate_titanic(200)
    assert df.shape == (200, 8)


def test_generate_titanic_columns():
    df = generate_titanic(100)
    cols = {"pclass", "age", "sex", "fare", "sibsp", "parch", "embarked", "survived"}
    assert cols.issubset(set(df.columns))


def test_generate_titanic_survived_binary():
    df = generate_titanic(500)
    assert set(df["survived"].unique()).issubset({0, 1})
