
from fastapi.testclient import TestClient

from app.main import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200
        assert "API is running" in response.json()["message"]


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["api_running"] is True
        assert response.json()["model_loaded"] is True


def test_predict():
    with TestClient(app) as client:
        data = {
            "trip_hours": 5.5,
            "distance_from_shore_km": 12.0,
            "crew": 4,
            "wave_height_m": 1.2,
            "water_temp_c": 25.5,
            "net_length_m": 812.0,
            "boat": "Trawler",
            "harbour": "North Jetty"
        }

        response = client.post("/predict", json=data)

        assert response.status_code == 200
        assert "predicted_catch_kg" in response.json()
        assert response.json()["unit"] == "kg"


def test_invalid_input():
    with TestClient(app) as client:
        data = {
            "trip_hours": -5,
            "distance_from_shore_km": 12.0,
            "crew": 4,
            "wave_height_m": 1.2,
            "water_temp_c": 25.5,
            "net_length_m": 812.0,
            "boat": "Trawler",
            "harbour": "North Jetty"
        }

        response = client.post("/predict", json=data)

        assert response.status_code == 422


def test_missing_model_returns_503():
    with TestClient(app) as client:
        original_model = app.state.model

        try:
            app.state.model = None

            data = {
                "trip_hours": 5.5,
                "distance_from_shore_km": 12.0,
                "crew": 4,
                "wave_height_m": 1.2,
                "water_temp_c": 25.5,
                "net_length_m": 812.0,
                "boat": "Trawler",
                "harbour": "North Jetty"
            }

            response = client.post("/predict", json=data)

            assert response.status_code == 503

        finally:
            app.state.model = original_model
