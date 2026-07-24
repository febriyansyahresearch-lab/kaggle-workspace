import numpy as np
from projects.iris.src.train import load_model
from projects.iris.src.preprocess import clean_data, encode_categorical, scale_features


def predict_new(model, features: np.ndarray) -> np.ndarray:
    return model.predict(features)


def predict_proba_new(model, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)