import joblib
import numpy as np
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Iris Classifier API", version="1.0.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
model = None


class Features(BaseModel):
    data: list[list[float]]


class Prediction(BaseModel):
    prediction: list
    probability: list[float] | None = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        model = None


@app.get("/health")
def health():
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array(features.data)
    preds = model.predict(X)
    proba = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None
    return Prediction(prediction=preds.tolist(), probability=proba)
