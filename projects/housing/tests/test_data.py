import pytest
from projects.housing.src.data import generate_housing


def test_generate_housing_shape():
    df = generate_housing(200)
    assert df.shape == (200, 9)


def test_generate_housing_columns():
    df = generate_housing(100)
    cols = {"med_income", "house_age", "avg_rooms", "avg_bedrooms", "population", "avg_occupancy", "latitude", "longitude", "price"}
    assert cols.issubset(set(df.columns))


def test_generate_housing_price_positive():
    df = generate_housing(500)
    assert (df["price"] > 0).all()
