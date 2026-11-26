
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ---------------- LOGGING CONFIGURATION ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- MODEL CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model_v2.joblib"


# ---------------- INPUT VALIDATION ----------------

class PredictionInput(BaseModel):
    trip_hours: float = Field(ge=0)
    distance_from_shore_km: float = Field(ge=0)
    crew: int = Field(ge=1)
    wave_height_m: float = Field(ge=0)
    water_temp_c: float
    net_length_m: float = Field(ge=0)

    boat: str = Field(min_length=1)
    harbour: str = Field(min_length=1)


# ---------------- MODEL LOADING ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fishing Boat Catch Prediction API")

    try:
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("V2 model loaded successfully from %s", MODEL_PATH)

    except FileNotFoundError:
        app.state.model = None
        logger.error("Model file not found: %s", MODEL_PATH)

    except Exception:
        app.state.model = None
        logger.exception("Failed to load the prediction model")

    yield

    logger.info("Shutting down Fishing Boat Catch Prediction API")


# ---------------- CREATE FASTAPI APP ----------------

app = FastAPI(
    title="Fishing Boat Catch Prediction API",
    description="Predict fishing boat catch quantity using the trained V2 ML model.",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------- VALIDATION ERROR LOGGING ----------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    logger.warning(
        "Request validation failed for %s %s",
        request.method,
        request.url.path
    )

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


# ---------------- HOME ENDPOINT ----------------

@app.get("/")
def home():
    return {
        "message": "Fishing Boat Catch Prediction API is running!"
    }


# ---------------- HEALTH ENDPOINT ----------------

@app.get("/health")
def health():
    model_loaded = app.state.model is not None

    return {
        "status": "healthy" if model_loaded else "degraded",
        "api_running": True,
        "model_loaded": model_loaded
    }


# ---------------- PREDICTION ENDPOINT ----------------

@app.post("/predict")
def predict(data: PredictionInput, request: Request):

    logger.info("Received prediction request")

    model = request.app.state.model

    if model is None:
        logger.error("Prediction unavailable because model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable"
        )

    try:
        input_data = pd.DataFrame([{
            "trip_hours": data.trip_hours,
            "distance_from_shore_km": data.distance_from_shore_km,
            "crew": data.crew,
            "wave_height_m": data.wave_height_m,
            "water_temp_c": data.water_temp_c,
            "net_length_m": data.net_length_m,
            "boat": data.boat,
            "harbour": data.harbour
        }])

        prediction = model.predict(input_data)

        predicted_catch = float(prediction[0])

        logger.info("Prediction completed successfully")

        return {
            "predicted_catch_kg": round(predicted_catch, 2),
            "unit": "kg"
        }

    except Exception:
        logger.exception("Unexpected error during prediction")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the prediction"
        )
