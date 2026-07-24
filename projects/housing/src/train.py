import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from projects.housing.src.data import generate_housing

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_models() -> dict:
    df = generate_housing()
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models["random_forest"], os.path.join(MODEL_DIR, "housing_model.joblib"))
    return results


if __name__ == "__main__":
    results = train_models()
    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
