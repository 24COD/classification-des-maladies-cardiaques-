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
"""

import json
import pickle
import io
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 0. Chargement des modèles et métadonnées au démarrage
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent.parent / "models"

_DEFAULT_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

def _load_joblib(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"{label} introuvable à '{path}'.")
    return joblib.load(path)

def _load_pickle(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"{label} introuvable à '{path}'.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


pipeline_binary = None
pipeline_multiclass = None
METADATA = None
ADJUSTED_THRESHOLD = 0.50  
FEATURES = _DEFAULT_FEATURES
_startup_error: Optional[str] = None

try:
    # ── Modèle binaire ────────────────────────────────────────────────
    try:
        pipeline_binary = _load_pickle(MODELS_DIR / "MEILLEUR_MODELE_BINAIRE.pkl", "Meilleur modèle binaire")
        print("Modele binaire charge (notebook/pkl)")
    except Exception as _pkl_err:
        print(f"Echec PKL ({_pkl_err}), tentative JOBLIB...")
        pipeline_binary = _load_joblib(MODELS_DIR / "model_binary.joblib", "Modèle binaire (joblib)")
        print("Modele binaire charge (joblib)")

    # ── Modèle multiclasse ────────────────────────────────────────────────
    pipeline_multiclass = _load_joblib(MODELS_DIR / "model_multiclass.joblib", "Modèle multiclasse")
    print("Modele multiclasse charge")

    # ── Métadonnées ───────────────────────────────────────────────────────
    _meta_path = MODELS_DIR / "metadata.json"
    if not _meta_path.exists():
        raise RuntimeError(f"metadata.json introuvable à '{_meta_path}'.")
    with open(_meta_path, encoding="utf-8") as fh:
        METADATA = json.load(fh)

    # ── Seuil ajusté ──────────────────────────────────────────────────────
    _seuil_path = MODELS_DIR / "seuil_retenu.json"
    if _seuil_path.exists():
        with open(_seuil_path, encoding="utf-8") as fh:
            seuil_data = json.load(fh)
        ADJUSTED_THRESHOLD = float(seuil_data.get("seuil_retenu", 0.50))
    else:
        ADJUSTED_THRESHOLD = float(METADATA["binary_model"].get("adjusted_threshold", 0.50))

    FEATURES = METADATA.get("features", _DEFAULT_FEATURES)

except Exception as _exc:
    _startup_error = str(_exc)
    print(f"CRITIQUE - Erreur au chargement des modeles : {_startup_error}")


# ---------------------------------------------------------------------------
# 1. Schémas Pydantic
# ---------------------------------------------------------------------------

class PatientFeatures(BaseModel):
    # Champs REQUIS
    age: float = Field(..., ge=0, le=120, description="Âge")
    sex: float = Field(..., ge=0, le=1, description="Sexe")
    cp:  float = Field(..., ge=0, le=4, description="Douleur thoracique")

    # Champs OPTIONNELS
    trestbps: Optional[float] = Field(None, ge=0)
    chol:     Optional[float] = Field(None, ge=0)
    fbs:      Optional[float] = Field(None, ge=0, le=1)
    restecg:  Optional[float] = Field(None, ge=0, le=2)
    thalach:  Optional[float] = Field(None, ge=0)
    exang:    Optional[float] = Field(None, ge=0, le=1)
    oldpeak:  Optional[float] = Field(None)
    slope:    Optional[float] = Field(None, ge=0, le=3)
    ca:       Optional[float] = Field(None, ge=0, le=4)
    thal:     Optional[float] = Field(None, ge=0, le=7)

    @field_validator("trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", mode="before")
    @classmethod
    def nan_to_none(cls, v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

class BatchRequest(BaseModel):
    patients: List[PatientFeatures] = Field(..., min_length=1)

# ---------------------------------------------------------------------------
# 2. Réponses
# ---------------------------------------------------------------------------

class BinaryPrediction(BaseModel):
    prediction_label:    str
    prediction_code:     int
    prediction_adjusted: int
    probability_disease: float
    threshold_default:   float = Field(0.50)
    threshold_adjusted:  float

class MulticlassPrediction(BaseModel):
    prediction_code: int
    probabilities:   List[float]

class BatchBinaryResponse(BaseModel):
    predictions: List[BinaryPrediction]

class BatchMulticlassResponse(BaseModel):
    predictions: List[MulticlassPrediction]

# ---------------------------------------------------------------------------
# 3. Helpers (SÉCURISÉS POUR L'IA)
# ---------------------------------------------------------------------------

def _check_models_loaded():
    if pipeline_binary is None or pipeline_multiclass is None:
        raise HTTPException(status_code=503, detail=f"Modèles non chargés : {_startup_error}")

def _patients_to_df(patients: List[PatientFeatures]) -> pd.DataFrame:
    """Convertit les requêtes JSON en DataFrame propre pour Scikit-Learn."""
    rows = [p.model_dump() for p in patients]
    df = pd.DataFrame(rows, columns=FEATURES)
    
    # SECURITÉ : Assure que les "None" envoyés en JSON redeviennent des "NaN" (pour l'imputer de l'IA)
    df.fillna(value=np.nan, inplace=True)
    # SECURITÉ : Force le format mathématique pour éviter les crashs de type "object"
    df = df.astype(float)
    return df

def _predict_binary(df: pd.DataFrame):
    all_proba = pipeline_binary.predict_proba(df)
    proba = all_proba[:, 0] if all_proba.shape[1] < 2 else all_proba[:, 1]
    pred_default  = (proba >= 0.50).astype(int)
    pred_adjusted = (proba >= ADJUSTED_THRESHOLD).astype(int)
    return proba, pred_default, pred_adjusted

def _predict_multiclass(df: pd.DataFrame):
    pred  = pipeline_multiclass.predict(df)
    proba = pipeline_multiclass.predict_proba(df)
    return pred, proba

# ---------------------------------------------------------------------------
# 4. Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="CardioRisk API", version="1.1.0")

@app.get("/", tags=["Info"])
def root():
    return {"message": "CardioRisk API en ligne", "version": "1.1.0"}

@app.get("/health", tags=["Info"])
def health():
    if pipeline_binary is None or pipeline_multiclass is None:
        raise HTTPException(status_code=503, detail=f"Modèles non chargés : {_startup_error}")
    return {"status": "ok", "adjusted_threshold": ADJUSTED_THRESHOLD}

# ---------------------------------------------------------------------------
# 5. Endpoints Individuels
# ---------------------------------------------------------------------------

@app.post("/predict/binary", response_model=BinaryPrediction, tags=["Prédiction"])
def predict_binary(patient: PatientFeatures):
    _check_models_loaded()
    df = _patients_to_df([patient])
    proba, pred_default, pred_adjusted = _predict_binary(df)
    return BinaryPrediction(
        prediction_label    = "disease" if pred_default[0] == 1 else "healthy",
        prediction_code     = int(pred_default[0]),
        prediction_adjusted = int(pred_adjusted[0]),
        probability_disease = round(float(proba[0]), 4),
        threshold_adjusted  = ADJUSTED_THRESHOLD,
    )

@app.post("/predict/multiclass", response_model=MulticlassPrediction, tags=["Prédiction"])
def predict_multiclass(patient: PatientFeatures):
    _check_models_loaded()
    df = _patients_to_df([patient])
    pred, proba = _predict_multiclass(df)
    return MulticlassPrediction(
        prediction_code = int(pred[0]),
        probabilities   = [round(float(p), 4) for p in proba[0]],
    )

# ---------------------------------------------------------------------------
# 6. Endpoints Batch (Prédiction en masse)
# ---------------------------------------------------------------------------

@app.post("/predict/binary/batch", response_model=BatchBinaryResponse, tags=["Prédiction (batch)"])
def predict_binary_batch(request: BatchRequest):
    _check_models_loaded()
    df = _patients_to_df(request.patients)
    proba, pred_default, pred_adjusted = _predict_binary(df)

    results = []
    for i in range(len(request.patients)):
        results.append(BinaryPrediction(
            prediction_label    = "disease" if pred_default[i] == 1 else "healthy",
            prediction_code     = int(pred_default[i]),
            prediction_adjusted = int(pred_adjusted[i]),
            probability_disease = round(float(proba[i]), 4),
            threshold_adjusted  = ADJUSTED_THRESHOLD,
        ))
    return BatchBinaryResponse(predictions=results)

@app.post("/predict/multiclass/batch", response_model=BatchMulticlassResponse, tags=["Prédiction (batch)"])
def predict_multiclass_batch(request: BatchRequest):
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

# ---------------------------------------------------------------------------
# 7. Endpoints CSV (Présents par sécurité, même si non utilisés par la nouvelle UI)
# ---------------------------------------------------------------------------

@app.post("/predict/binary/batch/csv", response_model=BatchBinaryResponse, tags=["Prédiction (batch)"])
async def predict_binary_batch_csv(file: UploadFile = File(...)):
    _check_models_loaded()
    try:
        content = await file.read()
        csv_text = content.decode("utf-8-sig")
        detected_sep = ";" if ";" in csv_text.splitlines()[0] else ","
        df = pd.read_csv(io.StringIO(csv_text), sep=detected_sep)
        df.columns = df.columns.str.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture : {e}")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing: raise HTTPException(status_code=400, detail=f"Colonnes manquantes : {missing}")

    df = df[FEATURES]
    df.fillna(value=np.nan, inplace=True)
    df = df.astype(float)
    
    proba, pred_default, pred_adjusted = _predict_binary(df)
    results = [
        BinaryPrediction(
            prediction_label="disease" if pred_default[i] == 1 else "healthy",
            prediction_code=int(pred_default[i]),
            prediction_adjusted=int(pred_adjusted[i]),
            probability_disease=round(float(proba[i]), 4),
            threshold_adjusted=ADJUSTED_THRESHOLD,
        ) for i in range(len(df))
    ]
    return BatchBinaryResponse(predictions=results)

@app.post("/predict/multiclass/batch/csv", response_model=BatchMulticlassResponse, tags=["Prédiction (batch)"])
async def predict_multiclass_batch_csv(file: UploadFile = File(...)):
    _check_models_loaded()
    try:
        content = await file.read()
        csv_text = content.decode("utf-8-sig")
        detected_sep = ";" if ";" in csv_text.splitlines()[0] else ","
        df = pd.read_csv(io.StringIO(csv_text), sep=detected_sep)
        df.columns = df.columns.str.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture : {e}")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing: raise HTTPException(status_code=400, detail=f"Colonnes manquantes : {missing}")

    df = df[FEATURES]
    df.fillna(value=np.nan, inplace=True)
    df = df.astype(float)

    pred, proba = _predict_multiclass(df)
    results = [
        MulticlassPrediction(
            prediction_code=int(pred[i]),
            probabilities=[round(float(p), 4) for p in proba[i]],
        ) for i in range(len(df))
    ]
    return BatchMulticlassResponse(predictions=results)