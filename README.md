# SCSE3040 MLOps Mini Project

## Group Details
- Group ID: G081
- Problem ID: MP149
- Problem: Fishing Boat Catch (Patna)
- Faculty: Dhananjay Kumar
- Batch: 80

## Team Members
| Enrolment Number | Name |
|---|---|
| S24CSEU2004 | Dhanishka Agarwal |
| S24CSEU2285 | Daksh Pratap Singh |
| S24CSEU2030 | Rudraksh Singh |
| S24CSEU2005 | Agam Bhatia |

## Project Objective
Build an end-to-end MLOps pipeline to predict fish catch in kilograms (catch_kg) for fishing boat trips.

## Evaluation Metric
Mean Absolute Error (MAE).

## Technology Stack
- Python
- DVC
- Git and GitHub
- Scikit-learn
- MLflow
- FastAPI
- Docker
- Kubernetes
- Pytest

## FastAPI Prediction Service

The project provides a REST API to predict fishing boat catch quantity using the trained V2 machine learning model.

### Model Used
- Model file: `models/model_v2.joblib`
- Algorithm: Extra Trees Regressor
- Output: Predicted catch in kilograms (`catch_kg`)

### Start the API

Activate the virtual environment and run:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check whether the API is running |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict fishing boat catch |

### Interactive API Documentation

Open the following URL after starting the server:

`http://127.0.0.1:8000/docs`

### Sample Prediction Request

```json
{
  "trip_hours": 5.5,
  "distance_from_shore_km": 12.0,
  "crew": 4,
  "wave_height_m": 1.2,
  "water_temp_c": 25.5,
  "net_length_m": 812.0,
  "boat": "Trawler",
  "harbour": "North Jetty"
}
```

### Sample Response

```json
{
  "predicted_catch_kg": 263.16,
  "unit": "kg"
}
```

### Run Automated Tests

```bash
python -m pytest -v
```

The tests cover API endpoints, input validation, model availability, and model prediction.

