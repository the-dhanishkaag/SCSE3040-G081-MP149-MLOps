import joblib
import pandas as pd

def test_model_loads():
    model = joblib.load("models/model_v2.joblib")
    assert model is not None

def test_model_predicts():
    model = joblib.load("models/model_v2.joblib")
    data = pd.DataFrame([{
        "trip_hours": 12.0,
        "distance_from_shore_km": 30.0,
        "crew": 7,
        "wave_height_m": 1.5,
        "water_temp_c": 27.0,
        "net_length_m": 800.0,
        "boat": "Type 1",
        "harbour": "North Jetty"
    }])
    prediction = model.predict(data)
    assert len(prediction) == 1
    assert prediction[0] >= 0
