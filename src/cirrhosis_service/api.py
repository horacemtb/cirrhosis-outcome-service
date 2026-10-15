"""Serve predictions and SHAP explanations from one loaded model."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cirrhosis_service.explain import explain_record, make_explainer
from cirrhosis_service.predict import load_model, predict_record
from cirrhosis_service.preprocessing import PatientData


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    app.state.model = load_model(path)
    app.state.explainer = make_explainer(app.state.model)
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


@app.post("/explain")
def explain(patient: PatientData, request: Request) -> dict:
    return explain_record(request.app.state.model, request.app.state.explainer, patient)
