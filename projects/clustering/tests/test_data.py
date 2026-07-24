import pytest
from projects.clustering.src.data import generate_customer_data


def test_generate_customer_data_shape():
    df = generate_customer_data(200)
    assert df.shape == (200, 4)


def test_generate_customer_data_columns():
    df = generate_customer_data(100)
    cols = {"age", "annual_income", "spending_score", "purchase_frequency"}
    assert cols.issubset(set(df.columns))


def test_generate_customer_data_valid_range():
    df = generate_customer_data(500)
    assert df["age"].between(18, 70).all()
    assert df["annual_income"].between(15, 150).all()
    assert df["spending_score"].between(1, 100).all()
