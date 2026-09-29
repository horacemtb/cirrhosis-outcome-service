"""Serve predictions from a model loaded once at startup."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cirrhosis_service.predict import load_model, predict_record
from cirrhosis_service.preprocessing import PatientData


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    app.state.model = load_model(path)
    yield


app = FastAPI(title="Cirrhosis Outcome Service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    # Avoid putting invalid numbers such as infinity into the JSON error response.
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientData, request: Request) -> dict:
    return predict_record(request.app.state.model, patient)
