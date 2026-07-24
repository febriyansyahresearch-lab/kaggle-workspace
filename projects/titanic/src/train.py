import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from projects.titanic.src.data import generate_titanic
from projects.titanic.src.features import engineer_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train(n_estimators: int = 100) -> dict:
    df = generate_titanic()
    df = engineer_features(df)
    X = df.drop(columns=["survived", "sibsp", "parch", "embarked"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "titanic_model.joblib"))
    return metrics


if __name__ == "__main__":
    metrics = train()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
