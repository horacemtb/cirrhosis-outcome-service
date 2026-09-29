"""Input fields and categorical encoding shared by training and prediction."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


class PatientData(BaseModel):
    """Patient's features, in the units used by the training CSV."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    Drug: Literal["Placebo", "D-penicillamine"]
    Age: int = Field(gt=0)
    Sex: Literal["F", "M"]
    Ascites: Literal["N", "Y"]
    Hepatomegaly: Literal["N", "Y"]
    Spiders: Literal["N", "Y"]
    Edema: Literal["N", "S", "Y"]
    Bilirubin: float = Field(gt=0)
    Cholesterol: float = Field(gt=0)
    Albumin: float = Field(gt=0)
    Copper: float = Field(gt=0)
    Alk_Phos: float = Field(gt=0)
    SGOT: float = Field(gt=0)
    Tryglicerides: float = Field(gt=0)
    Platelets: float = Field(gt=0)
    Prothrombin: float = Field(gt=0)
    Stage: int = Field(ge=1, le=4)


CATEGORIES = {
    "Drug": ["Placebo", "D-penicillamine"],
    "Sex": ["F", "M"],
    "Ascites": ["N", "Y"],
    "Hepatomegaly": ["N", "Y"],
    "Spiders": ["N", "Y"],
    "Edema": ["N", "S", "Y"]
}

NUMERIC_FEATURES = [
    "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
]


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the same input rules for CSV rows and API requests."""
    if frame.empty:
        raise ValueError("No rows to process.")

    rows = []
    for record in frame.to_dict(orient="records"):
        patient = PatientData.model_validate(record)
        rows.append(patient.model_dump())
    return pd.DataFrame(rows)


def make_preprocessor() -> ColumnTransformer:
    """Encode categories in a fixed order and keep numeric values unchanged."""
    return ColumnTransformer([
        ("categories", OrdinalEncoder(categories=list(CATEGORIES.values())), list(CATEGORIES)),
        ("stage", OrdinalEncoder(categories=[[1, 2, 3, 4]]), ["Stage"]),
        ("numeric", "passthrough", NUMERIC_FEATURES)
    ])
