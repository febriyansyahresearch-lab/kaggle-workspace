import numpy as np
import pandas as pd


def generate_housing(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    med_income = rng.lognormal(mean=np.log(5), sigma=0.5, size=n)
    house_age = rng.uniform(0, 50, size=n)
    avg_rooms = rng.normal(6, 2, size=n).clip(2, 15)
    avg_bedrooms = avg_rooms * rng.uniform(0.15, 0.25, size=n)
    population = rng.poisson(500, size=n)
    avg_occupancy = rng.uniform(2, 6, size=n)
    latitude = rng.uniform(32.5, 42.0, size=n)
    longitude = rng.uniform(-124, -114, size=n)

    price = (
        -2.5 * med_income
        + 0.3 * house_age
        + 3.0 * avg_rooms
        - 2.0 * avg_bedrooms
        + 0.001 * population
        + 0.5 * avg_occupancy
        + 50 * (latitude - 37)
        - 30 * (longitude + 119)
        + rng.normal(0, 15, size=n)
        + 200
    ).clip(50, 500)

    df = pd.DataFrame({
        "med_income": med_income,
        "house_age": house_age,
        "avg_rooms": avg_rooms,
        "avg_bedrooms": avg_bedrooms,
        "population": population,
        "avg_occupancy": avg_occupancy,
        "latitude": latitude,
        "longitude": longitude,
        "price": price,
    })
    return df.round(2)
