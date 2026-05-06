"""
api.py
======
API FastAPI pour les deux modèles de prédiction du risque cardiaque (UCI Heart Disease).

Endpoints disponibles :
  GET  /                        → accueil + liste des endpoints
  GET  /health                  → statut de l'API et des modèles
  POST /predict/binary          → prédiction binaire (0 = sain, 1 = malade)
  POST /predict/multiclass      → prédiction multiclasse (0 à 4)
  POST /predict/binary/batch    → prédiction binaire sur plusieurs patients
  POST /predict/multiclass/batch→ prédiction multiclasse sur plusieurs patients

Lancement :
  pip install fastapi uvicorn joblib scikit-learn pandas numpy
  python export_models.py          # génère models/ si pas encore fait
  uvicorn api:app --reload --port 8000

Documentation interactive :
  http://localhost:8000/docs       (Swagger UI)
  http://localhost:8000/redoc      (ReDoc)
"""

import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 0. Chargement des modèles et métadonnées au démarrage
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")

def _load_artifact(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(
            f"{label} introuvable à '{path}'. "
            "Lancez d'abord export_models.py pour générer les modèles."
        )
    return joblib.load(path)

try:
    pipeline_binary     = _load_artifact(MODELS_DIR / "model_binary.joblib",     "Modèle binaire")
    pipeline_multiclass = _load_artifact(MODELS_DIR / "model_multiclass.joblib", "Modèle multiclasse")
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as fh:
        METADATA = json.load(fh)
    ADJUSTED_THRESHOLD = float(METADATA["binary_model"].get("adjusted_threshold", 0.50))
    FEATURES = METADATA["features"]
except RuntimeError as exc:
    # Démarrage possible même sans modèles ; /health renverra 503
    pipeline_binary = pipeline_multiclass = METADATA = None
    ADJUSTED_THRESHOLD = 0.50
    FEATURES = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    _startup_error = str(exc)
else:
    _startup_error = None

# ---------------------------------------------------------------------------
# 1. Schémas Pydantic
# ---------------------------------------------------------------------------

class PatientFeatures(BaseModel):
    """Variables cliniques d'un patient. Les valeurs manquantes sont acceptées (None)."""

    age:      float = Field(..., ge=0, le=120,  description="Âge en années")
    sex:      float = Field(..., ge=0, le=1,    description="Sexe : 1 = homme, 0 = femme")
    cp:       float = Field(..., ge=0, le=4,    description="Type de douleur thoracique (0-4)")
    trestbps: Optional[float] = Field(None, ge=0, description="Tension artérielle au repos (mm Hg)")
    chol:     Optional[float] = Field(None, ge=0, description="Cholestérol sérique (mg/dl)")
    fbs:      Optional[float] = Field(None, ge=0, le=1,  description="Glycémie à jeun > 120 mg/dl (1 = vrai)")
    restecg:  Optional[float] = Field(None, ge=0, le=2,  description="Résultats ECG au repos (0-2)")
    thalach:  Optional[float] = Field(None, ge=0, description="Fréquence cardiaque maximale atteinte")
    exang:    Optional[float] = Field(None, ge=0, le=1,  description="Angine induite par l'effort (1 = oui)")
    oldpeak:  Optional[float] = Field(None,       description="Dépression ST induite par l'effort")
    slope:    Optional[float] = Field(None, ge=0, le=3,  description="Pente du segment ST à l'effort (0-3)")
    ca:       Optional[float] = Field(None, ge=0, le=4,  description="Nb de vaisseaux colorés par fluoroscopie (0-4)")
    thal:     Optional[float] = Field(None, ge=0, le=7,  description="Thalassémie (3 = normal, 6 = défaut fixe, 7 = défaut réversible)")

    model_config = {"json_schema_extra": {
        "example": {
            "age": 54, "sex": 1, "cp": 2, "trestbps": 130, "chol": 246,
            "fbs": 0, "restecg": 0, "thalach": 173, "exang": 0,
            "oldpeak": 0.0, "slope": 2, "ca": None, "thal": None,
        }
    }}

    @field_validator("*", mode="before")
    @classmethod
    def nan_to_none(cls, v):
        """Convertit NaN flottant en None (compatibilité JSON)."""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class BatchRequest(BaseModel):
    patients: List[PatientFeatures] = Field(..., min_length=1, description="Liste de patients")


# ---------------------------------------------------------------------------
# 2. Réponses
# ---------------------------------------------------------------------------

class BinaryPrediction(BaseModel):
    prediction_label:     str   = Field(..., description="'healthy' ou 'disease'")
    prediction_code:      int   = Field(..., description="0 = sain, 1 = malade (seuil 0.50)")
    prediction_adjusted:  int   = Field(..., description="Prédiction au seuil ajusté (priorité recall)")
    probability_disease:  float = Field(..., description="Probabilité de maladie (classe 1)")
    threshold_default:    float = Field(0.50)
    threshold_adjusted:   float = Field(...)


class MulticlassPrediction(BaseModel):
    prediction_code:      int         = Field(..., description="Classe prédite : 0 à 4")
    probabilities:        List[float] = Field(..., description="Probabilités par classe [0,1,2,3,4]")


class BatchBinaryResponse(BaseModel):
    predictions: List[BinaryPrediction]


class BatchMulticlassResponse(BaseModel):
    predictions: List[MulticlassPrediction]


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def _check_models_loaded():
    if pipeline_binary is None or pipeline_multiclass is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modèles non chargés. Raison : {_startup_error}",
        )


def _patients_to_df(patients: List[PatientFeatures]) -> pd.DataFrame:
    """Convertit une liste de PatientFeatures en DataFrame avec les bonnes colonnes."""
    rows = [p.model_dump() for p in patients]
    df = pd.DataFrame(rows, columns=FEATURES)
    return df


def _predict_binary(df: pd.DataFrame):
    proba = pipeline_binary.predict_proba(df)[:, 1]
    pred_default  = (proba >= 0.50).astype(int)
    pred_adjusted = (proba >= ADJUSTED_THRESHOLD).astype(int)
    return proba, pred_default, pred_adjusted


def _predict_multiclass(df: pd.DataFrame):
    pred   = pipeline_multiclass.predict(df)
    proba  = pipeline_multiclass.predict_proba(df)
    return pred, proba


# ---------------------------------------------------------------------------
# 4. Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "API de prédiction du risque de maladie cardiaque basée sur le dataset UCI. "
        "Deux modèles sont disponibles : classification **binaire** (sain / malade) "
        "et classification **multiclasse** (sévérité 0-4)."
    ),
    version="1.0.0",
)


@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Heart Disease Prediction API — v1.0.0",
        "endpoints": {
            "health":                 "GET  /health",
            "predict_binary":         "POST /predict/binary",
            "predict_multiclass":     "POST /predict/multiclass",
            "predict_binary_batch":   "POST /predict/binary/batch",
            "predict_multiclass_batch": "POST /predict/multiclass/batch",
            "docs":                   "GET  /docs",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    """Vérifie que l'API et les modèles sont opérationnels."""
    if pipeline_binary is None or pipeline_multiclass is None:
        raise HTTPException(status_code=503, detail=f"Modèles non chargés : {_startup_error}")
    return {
        "status": "ok",
        "binary_model":     METADATA["binary_model"]["model"],
        "multiclass_model": METADATA["multiclass_model"]["model"],
        "binary_roc_auc":   METADATA["binary_model"]["test_roc_auc"],
        "multiclass_roc_auc": METADATA["multiclass_model"]["test_roc_auc"],
        "adjusted_threshold": ADJUSTED_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 5. Endpoints de prédiction — individuel
# ---------------------------------------------------------------------------

@app.post("/predict/binary", response_model=BinaryPrediction, tags=["Prédiction"])
def predict_binary(patient: PatientFeatures):
    """
    Prédit le **risque binaire** (sain = 0 / malade = 1) pour un seul patient.

    Renvoie :
    - la prédiction au **seuil par défaut (0.50)** ;
    - la prédiction au **seuil ajusté** (optimisé pour le recall).
    """
    _check_models_loaded()
    df = _patients_to_df([patient])
    proba, pred_default, pred_adjusted = _predict_binary(df)

    p = float(proba[0])
    return BinaryPrediction(
        prediction_label    = "disease" if pred_default[0] == 1 else "healthy",
        prediction_code     = int(pred_default[0]),
        prediction_adjusted = int(pred_adjusted[0]),
        probability_disease = round(p, 4),
        threshold_default   = 0.50,
        threshold_adjusted  = ADJUSTED_THRESHOLD,
    )


@app.post("/predict/multiclass", response_model=MulticlassPrediction, tags=["Prédiction"])
def predict_multiclass(patient: PatientFeatures):
    """
    Prédit la **sévérité de la maladie** (0 = sain, 1-4 = degrés croissants) pour un seul patient.
    """
    _check_models_loaded()
    df = _patients_to_df([patient])
    pred, proba = _predict_multiclass(df)

    return MulticlassPrediction(
        prediction_code = int(pred[0]),
        probabilities   = [round(float(p), 4) for p in proba[0]],
    )


# ---------------------------------------------------------------------------
# 6. Endpoints de prédiction — batch
# ---------------------------------------------------------------------------

@app.post("/predict/binary/batch", response_model=BatchBinaryResponse, tags=["Prédiction (batch)"])
def predict_binary_batch(request: BatchRequest):
    """
    Prédit le **risque binaire** pour une liste de patients en une seule requête.
    """
    _check_models_loaded()
    df = _patients_to_df(request.patients)
    proba, pred_default, pred_adjusted = _predict_binary(df)

    results = []
    for i in range(len(request.patients)):
        p = float(proba[i])
        results.append(BinaryPrediction(
            prediction_label    = "disease" if pred_default[i] == 1 else "healthy",
            prediction_code     = int(pred_default[i]),
            prediction_adjusted = int(pred_adjusted[i]),
            probability_disease = round(p, 4),
            threshold_default   = 0.50,
            threshold_adjusted  = ADJUSTED_THRESHOLD,
        ))

    return BatchBinaryResponse(predictions=results)


@app.post("/predict/multiclass/batch", response_model=BatchMulticlassResponse, tags=["Prédiction (batch)"])
def predict_multiclass_batch(request: BatchRequest):
    """
    Prédit la **sévérité** pour une liste de patients en une seule requête.
    """
    _check_models_loaded()
    df = _patients_to_df(request.patients)
    pred, proba = _predict_multiclass(df)

    results = []
    for i in range(len(request.patients)):
        results.append(MulticlassPrediction(
            prediction_code = int(pred[i]),
            probabilities   = [round(float(p), 4) for p in proba[i]],
        ))

    return BatchMulticlassResponse(predictions=results)
