import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "heart_disease_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_scaler_loads():
    scaler = joblib.load(SCALER_PATH)
    assert scaler is not None

def test_prediction_pipeline():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sample_input = np.zeros((1, 13))
    scaled = scaler.transform(sample_input)
    prediction = model.predict(scaled)

    assert prediction.shape == (1,)