from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_model(model, filename: str = "model.joblib"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(filename: str = "model.joblib"):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)
