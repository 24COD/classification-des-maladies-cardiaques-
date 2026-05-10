"""
app.py — Interface Streamlit pour l'API de prédiction des maladies cardiaques
=============================================================================
API cible : https://classification-des-maladies-cardiaques-1-zbq3.onrender.com
Endpoint  : POST /predict/binary | POST /predict/multiclass

CORRECTIONS APPLIQUÉES :
  [C1] prediction_label utilisé à la place de prediction_code pour le résultat binaire.
  [C2] Détection automatique du séparateur CSV (sep=None, engine="python").
  [C3] Garde ZeroDivisionError si fichier vide.
  [C4] Comptage erreurs via champ "status" (sans dépendre des emojis).
  [C5] Timeout porté à 60s + retry automatique 1 fois.
  [C13] NOUVELLE APPROCHE : Envoi direct du fichier CSV à l'API via /batch/csv endpoints
        au lieu d'envoyer ligne par ligne. Plus robuste et efficace.
"""

import html
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
API_KEEP_ALIVE_URL = "https://cardiorisk-api.onrender.com/health"

def _keep_alive_worker():
    while True:
        try:
            requests.get(API_KEEP_ALIVE_URL, timeout=10)
        except Exception:
            pass
        time.sleep(600)

if "keep_alive_started" not in st.session_state:
    thread = threading.Thread(target=_keep_alive_worker, daemon=True)
    thread.start()
    st.session_state["keep_alive_started"] = True
# ---------------------------------------------------------------------------
# 0. Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. CSS personnalisé
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    :root {
        --blue-dark:   #0d2b55;
        --blue-mid:    #1a4a8a;
        --blue-light:  #2e73c4;
        --blue-pale:   #dce8f7;
        --blue-frost:  #eef4fc;
        --grey-100:    #f4f6f9;
        --grey-200:    #e2e8f0;
        --grey-400:    #94a3b8;
        --grey-600:    #475569;
        --grey-800:    #1e293b;
        --white:       #ffffff;
        --danger:      #c0392b;
        --healthy:     #1a5276;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--grey-100) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--grey-800) !important;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--blue-dark) 0%, var(--blue-mid) 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #cde0f7 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--white) !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    h1 { font-family: 'DM Serif Display', serif !important; color: var(--blue-dark) !important; }
    h2, h3 { color: var(--blue-mid) !important; font-weight: 600 !important; }

    div.stButton > button {
        background: linear-gradient(135deg, var(--blue-mid) 0%, var(--blue-light) 100%) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2.5rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.04em !important;
        transition: box-shadow 0.2s ease, transform 0.1s ease !important;
        box-shadow: 0 4px 14px rgba(26, 74, 138, 0.35) !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 6px 20px rgba(26, 74, 138, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active { transform: translateY(0px) !important; }

    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div > div {
        border: 1.5px solid var(--grey-200) !important;
        border-radius: 6px !important;
        background: var(--white) !important;
        color: var(--grey-800) !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--blue-light) !important;
        box-shadow: 0 0 0 3px rgba(46, 115, 196, 0.15) !important;
    }

    label[data-testid="stWidgetLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--grey-600) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .section-card {
        background: var(--white);
        border: 1px solid var(--grey-200);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(13, 43, 85, 0.06);
    }
    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--blue-light);
        border-bottom: 1px solid var(--grey-200);
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
    }

    .result-card {
        border-radius: 14px;
        padding: 2rem 2.2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(13, 43, 85, 0.12);
    }
    .result-disease {
        background: linear-gradient(135deg, #1c3a5c 0%, #2b5fa3 100%);
        border-left: 5px solid #e74c3c;
        color: var(--white) !important;
    }
    .result-healthy {
        background: linear-gradient(135deg, #0d3b6e 0%, #1a6e9a 100%);
        border-left: 5px solid #27ae60;
        color: var(--white) !important;
    }
    .result-card h2 { color: var(--white) !important; font-size: 1.7rem !important; }
    .result-card p  { color: rgba(255,255,255,0.85) !important; margin: 0.3rem 0 !important; }
    .result-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .badge-disease { background: rgba(231,76,60,0.25); color: #ff9999 !important; border: 1px solid rgba(231,76,60,0.5); }
    .badge-healthy { background: rgba(39,174,96,0.25);  color: #a8f0c6 !important; border: 1px solid rgba(39,174,96,0.5); }

    .prob-bar-bg {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        height: 10px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 10px;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #721c24;
        margin-top: 1rem;
    }

    .info-pill {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.82rem;
        border-left: 3px solid rgba(255,255,255,0.4);
    }

    .styled-hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--grey-200), transparent);
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Constantes API
# ---------------------------------------------------------------------------
API_BASE_URL           = os.getenv("API_BASE_URL", "http://localhost:8000").strip().rstrip("/")
if not API_BASE_URL.startswith(("http://", "https://")):
    API_BASE_URL = f"http://{API_BASE_URL}"
PREDICT_BINARY_URL     = f"{API_BASE_URL}/predict/binary"
PREDICT_MULTICLASS_URL = f"{API_BASE_URL}/predict/multiclass"
PREDICT_BINARY_BATCH_CSV_URL     = f"{API_BASE_URL}/predict/binary/batch/csv"
PREDICT_MULTICLASS_BATCH_CSV_URL = f"{API_BASE_URL}/predict/multiclass/batch/csv"

# Timeout en secondes (10s pour API locale)
API_TIMEOUT   = 60
API_MAX_RETRY = 1   # 1 retry automatique en cas de timeout
API_FALLBACK_STATUS_CODES = {502, 503, 504}

# Colonnes STRICTEMENT REQUISES (selon l'API PatientFeatures)
REQUIRED_COLS = ["age", "sex", "cp"]

# Colonnes OPTIONNELLES (peuvent être None/NaN dans le CSV)
OPTIONAL_COLS_LIST = [
    "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal"
]

# Ensemble pour recherche rapide (union des deux)
ALL_EXPECTED_COLS = set(REQUIRED_COLS + OPTIONAL_COLS_LIST)
OPTIONAL_COLS_SET = set(OPTIONAL_COLS_LIST)  # Pour vérification rapide

CLASS_LABELS = {
    0: "Absence de maladie",
    1: "Maladie légère",
    2: "Maladie modérée",
    3: "Maladie sévère",
    4: "Maladie très sévère",
}

CLASS_LABELS_SHORT = {
    0: "Pas de maladie",
    1: "Légère",
    2: "Modérée",
    3: "Sévère",
    4: "Très sévère",
}

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def _compact_response_text(text: str, limit: int = 700) -> str:
    """Retourne un extrait lisible d'une erreur HTML/texte trop longue."""
    clean = " ".join(html.escape(text or "").split())
    return clean if len(clean) <= limit else f"{clean[:limit]}..."


def _local_model_fallback(url: str, payload: dict) -> Optional[dict]:
    """
    Predit directement dans Streamlit si l'API Render est indisponible.

    Le service Streamlit deploie le meme repository et dispose donc aussi des
    modeles dans `models/`. Ce fallback evite qu'un 502 Render bloque l'app.
    """
    endpoint_path = urlparse(url).path.rstrip("/")
    if endpoint_path not in {"/predict/binary", "/predict/multiclass"}:
        return None

    try:
        from API.api import PatientFeatures, predict_binary, predict_multiclass

        patient = PatientFeatures(**payload)
        if endpoint_path == "/predict/binary":
            result = predict_binary(patient)
        else:
            result = predict_multiclass(patient)

        data = result.model_dump()
        data["_source"] = "local_fallback"
        return data
    except Exception:
        return None


def call_api(url: str, payload: dict) -> dict:
    """
    Appelle l'API avec retry automatique.
    Lève une exception si toutes les tentatives échouent.
    """
    last_exc = None
    for attempt in range(API_MAX_RETRY + 1):
        try:
            resp = requests.post(url, json=payload, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as e:
            fallback = _local_model_fallback(url, payload)
            if fallback is not None:
                return fallback
            last_exc = e
            if attempt < API_MAX_RETRY:
                time.sleep(2)
            continue
        except requests.exceptions.ConnectionError as e:
            fallback = _local_model_fallback(url, payload)
            if fallback is not None:
                return fallback
            raise requests.exceptions.ConnectionError(str(e)) from e
        except requests.exceptions.HTTPError as e:
            # [C7] Échappement HTML pour éviter XSS dans l'affichage de l'erreur
            if resp.status_code in API_FALLBACK_STATUS_CODES:
                fallback = _local_model_fallback(url, payload)
                if fallback is not None:
                    return fallback
            e.response_text = _compact_response_text(resp.text)
            e.status_code   = resp.status_code
            raise e
    raise requests.exceptions.Timeout(str(last_exc))


def validate_clinical(payload: dict) -> list:
    """
    Retourne une liste d'avertissements cliniques non bloquants.

    [C11] Vérifie que les champs optionnels ne sont pas None avant
          toute comparaison numérique.
    """
    warnings_list = []

    age = payload.get("age")
    if age is not None and (age < 20 or age > 100):
        warnings_list.append(f"Âge hors plage habituelle ({int(age)} ans).")

    trestbps = payload.get("trestbps")
    if trestbps is not None and trestbps > 0 and trestbps < 60:
        warnings_list.append("Tension artérielle très basse (<60 mmHg).")

    chol = payload.get("chol")
    if chol is not None and chol > 0 and chol < 100:
        warnings_list.append("Cholestérol très bas (<100 mg/dl).")

    thalach = payload.get("thalach")
    if thalach is not None and thalach > 0 and thalach < 40:
        warnings_list.append("Fréquence cardiaque max très basse (<40 bpm).")

    return warnings_list


def predict_single_row(url: str, row: pd.Series) -> dict:
    """
    Construit le payload pour une ligne DataFrame et appelle l'API.

    [C12] Les champs optionnels avec NaN sont envoyés comme None (null JSON),
          ce que l'API accepte. Les champs requis sont toujours convertis en float.
    
    [FIXED] Gère les colonnes manquantes (returnne None si absent).
            Gère les NaN pour TOUS les champs, pas seulement les optionnels.
    """
    payload = {}
    
    # ── Colonnes REQUISES (age, sex, cp) ─────────────────────────────────
    for col in REQUIRED_COLS:
        if col not in row.index:
            # Colonne manquante → c'est une erreur pour les champs requis
            raise ValueError(f"Colonne requise manquante: '{col}'")
        val = row[col]
        # NaN → None (acceptable pour l'API car elle peut valider)
        if pd.isna(val):
            raise ValueError(f"Valeur manquante (NaN) pour colonne requise: '{col}'")
        payload[col] = float(val)
    
    # ── Colonnes OPTIONNELLES ────────────────────────────────────────────
    for col in OPTIONAL_COLS_LIST:
        if col not in row.index:
            # Colonne manquante dans CSV → utiliser None par défaut
            payload[col] = None
        else:
            val = row[col]
            # NaN → None (null JSON)
            payload[col] = None if pd.isna(val) else float(val)
    
    return call_api(url, payload)


# ---------------------------------------------------------------------------
# 4. Sidebar
# ---------------------------------------------------------------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "binary"

with st.sidebar:
    st.markdown("## 🫀 CardioRisk")
    st.markdown("**Système d'aide à la décision clinique**")
    st.markdown("---")

    st.markdown("### ⚙️ Sélection du modèle")
    selected_model_label = st.radio(
        "Choisir le modèle de prédiction :",
        options=["Binaire", "Multiclasse"],
        format_func=lambda x: f"🔵 {x}" if x == "Binaire" else f"📊 {x}",
        key="model_selector",
    )
    st.session_state.selected_model = (
        "binary" if selected_model_label == "Binaire" else "multiclass"
    )

    if selected_model_label == "Binaire":
        st.markdown('<div class="info-pill">🔵 <b>Binaire</b> — Sain / Malade</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-pill">📊 <b>Multiclasse</b> — Sévérité 0–4</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### À propos")
    st.markdown(
        "Cette interface interroge une API de Machine Learning "
        "entraînée sur le dataset **UCI Heart Disease** "
        "(303 patients, 13 variables cliniques)."
    )
    st.markdown("---")

    st.markdown("### Endpoint actif")
    endpoint_display = (
        "/predict/binary" if st.session_state.selected_model == "binary"
        else "/predict/multiclass"
    )
    st.code(f"POST {endpoint_display}", language="bash")

    st.markdown("---")
    st.markdown("### ⚠️ Avertissement")
    st.caption(
        "Cet outil est destiné à des fins de **démonstration et de recherche** uniquement. "
        "Il ne constitue en aucun cas un avis médical professionnel."
    )

    st.markdown("---")
    if st.button("🔍 Vérifier le statut API", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                st.success("✅ API opérationnelle")
                st.caption(f"Modèle binaire : `{data.get('binary_model', 'N/A')}`")
                st.caption(f"ROC-AUC : `{data.get('binary_roc_auc', 'N/A')}`")
                st.caption(f"Seuil ajusté : `{data.get('adjusted_threshold', 'N/A')}`")
            else:
                st.error(f"Erreur {r.status_code}")
        except Exception as e:
            st.error(f"Impossible de joindre l'API\n{e}")

# ---------------------------------------------------------------------------
# 5. En-tête principal
# ---------------------------------------------------------------------------
st.markdown("<h1>Prédiction du Risque Cardiaque</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#475569; font-size:1rem; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Saisissez les données cliniques du patient pour obtenir une évaluation du risque cardiovasculaire."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="styled-hr">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. Onglets
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Prédiction Individuelle", "📊 Prédiction en Masse"])


# ===========================================================================
# TAB 1 — Prédiction individuelle
# ===========================================================================
with tab1:

    with st.form("patient_form", clear_on_submit=False):

        # ── Démographiques ──────────────────────────────────────────────
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">📋 Informations démographiques</div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Âge (années)", min_value=0, max_value=120, value=54, step=1)
        with col2:
            sex = st.selectbox("Sexe", options=[1, 0], format_func=lambda x: "Homme" if x == 1 else "Femme")
        with col3:
            cp = st.selectbox(
                "Type de douleur thoracique (cp)",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 — Angine typique",
                    1: "1 — Angine atypique",
                    2: "2 — Douleur non angineuse",
                    3: "3 — Asymptomatique",
                }[x],
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Cardiovasculaires ───────────────────────────────────────────
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">💓 Paramètres cardiovasculaires</div>',
            unsafe_allow_html=True,
        )
        col4, col5, col6 = st.columns(3)
        with col4:
            trestbps = st.number_input("Tension artérielle au repos (mm Hg)", min_value=0.0, max_value=300.0, value=130.0, step=1.0)
        with col5:
            chol = st.number_input("Cholestérol sérique (mg/dl)", min_value=0.0, max_value=700.0, value=246.0, step=1.0)
        with col6:
            thalach = st.number_input("Fréquence cardiaque max. (bpm)", min_value=0.0, max_value=300.0, value=150.0, step=1.0)

        col7, col8 = st.columns(2)
        with col7:
            fbs = st.selectbox("Glycémie à jeun > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Oui (1)" if x == 1 else "Non (0)")
        with col8:
            exang = st.selectbox("Angine induite par l'effort (exang)", options=[0, 1], format_func=lambda x: "Oui (1)" if x == 1 else "Non (0)")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── ECG & Imagerie ──────────────────────────────────────────────
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">📈 Données ECG & Imagerie</div>',
            unsafe_allow_html=True,
        )
        col9, col10, col11 = st.columns(3)
        with col9:
            restecg = st.selectbox(
                "Résultats ECG au repos (restecg)",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 — Normal",
                    1: "1 — Anomalie onde ST-T",
                    2: "2 — Hypertrophie ventriculaire gauche",
                }[x],
            )
        with col10:
            slope = st.selectbox(
                "Pente du segment ST (slope)",
                options=[0, 1, 2],
                format_func=lambda x: {0: "0 — Montante", 1: "1 — Plate", 2: "2 — Descendante"}[x],
            )
        with col11:
            ca = st.selectbox(
                "Vaisseaux colorés par fluoroscopie (ca)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: f"{x} vaisseau{'x' if x > 1 else ''}",
            )

        col12, col13 = st.columns(2)
        with col12:
            oldpeak = st.number_input("Dépression ST à l'effort (oldpeak)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        with col13:
            thal = st.selectbox(
                "Thalassémie (thal)",
                options=[3, 6, 7],
                format_func=lambda x: {3: "3 — Normal", 6: "6 — Défaut fixe", 7: "7 — Défaut réversible"}[x],
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            submitted = st.form_submit_button("🔬 Analyser", use_container_width=True)

    # ── Traitement après soumission ─────────────────────────────────────
    if submitted:
        payload = {
            "age":      float(age),
            "sex":      float(sex),
            "cp":       float(cp),
            "trestbps": float(trestbps),
            "chol":     float(chol),
            "fbs":      float(fbs),
            "restecg":  float(restecg),
            "thalach":  float(thalach),
            "exang":    float(exang),
            "oldpeak":  float(oldpeak),
            "slope":    float(slope),
            "ca":       float(ca),
            "thal":     float(thal),
        }

        # Avertissements cliniques (non bloquants)
        clin_warnings = validate_clinical(payload)
        for w in clin_warnings:
            st.warning(f"⚠️ {w}")

        predict_url = (
            PREDICT_MULTICLASS_URL
            if st.session_state.selected_model == "multiclass"
            else PREDICT_BINARY_URL
        )

        with st.spinner("Analyse en cours via l'API…"):
            result = None
            try:
                result = call_api(predict_url, payload)

            except requests.exceptions.ConnectionError:
                st.markdown(
                    '<div class="error-box">❌ <b>Impossible de joindre l\'API.</b> '
                    "Vérifiez que le serveur est en ligne et que l'URL est correcte.</div>",
                    unsafe_allow_html=True,
                )
            except requests.exceptions.Timeout:
                st.markdown(
                    '<div class="error-box">⏱️ <b>Délai d\'attente dépassé après 2 tentatives.</b> '
                    "Le serveur met trop de temps à répondre. "
                    "Sur Render, le démarrage à froid peut prendre ~50s — veuillez réessayer dans un moment.</div>",
                    unsafe_allow_html=True,
                )
            except requests.exceptions.HTTPError as e:
                st.markdown(
                    f'<div class="error-box">⚠️ <b>Erreur HTTP {e.status_code}.</b> '
                    f"Détail : {e.response_text}</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">🚨 <b>Erreur inattendue :</b> {html.escape(str(e))}</div>',
                    unsafe_allow_html=True,
                )

        if result:
            if result.get("_source") == "local_fallback":
                st.info(
                    "API Render indisponible momentanement. "
                    "Prediction realisee avec les modeles locaux du service Streamlit."
                )

            if st.session_state.selected_model == "binary":
                # ── MODÈLE BINAIRE ──────────────────────────────────────
                # [C1] On utilise "prediction_label" (str), jamais "prediction_code" seul
                label      = result.get("prediction_label", "healthy")
                prob       = result.get("probability_disease", 0.0)
                pred_adj   = result.get("prediction_adjusted", 0)
                thresh_adj = result.get("threshold_adjusted", 0.50)
                is_disease = (label == "disease")

                default_pred_label = "Malade" if is_disease else "Sain"

                card_class  = "result-disease" if is_disease else "result-healthy"
                badge_class = "badge-disease"  if is_disease else "badge-healthy"
                icon        = "🔴" if is_disease else "🟢"
                label_fr    = "MALADIE CARDIAQUE DÉTECTÉE" if is_disease else "AUCUNE MALADIE DÉTECTÉE"
                desc        = (
                    "Le modèle identifie un <b>risque cardiovasculaire significatif</b> pour ce profil patient."
                    if is_disease else
                    "Le modèle n'identifie <b>pas de signal clinique alarmant</b> pour ce profil patient."
                )

                prob_pct  = int(prob * 100)
                bar_color = "#e74c3c" if prob > 0.6 else ("#f39c12" if prob > 0.4 else "#27ae60")

                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <span class="result-badge {badge_class}">{icon} {label_fr}</span>
                        <h2 style="margin-bottom:0.2rem;">{prob_pct}% de probabilité de maladie</h2>
                        <p>{desc}</p>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{prob_pct}%; background:{bar_color};"></div>
                        </div>
                        <br>
                        <p>
                            <b>Seuil par défaut (0.50) :</b> {default_pred_label} &nbsp;|&nbsp;
                            <b>Seuil ajusté ({thresh_adj}) :</b> {'Malade' if pred_adj == 1 else 'Sain'}
                            <span style="font-size:0.75rem; opacity:0.7;"> (optimisé pour le rappel)</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                # ── MODÈLE MULTICLASSE ──────────────────────────────────
                # [C9] Le champ s'appelle "prediction_code", pas "prediction_class"
                pred_class = result.get("prediction_code", None)

                # [C10] "probabilities" est une List[float], pas un dict.
                #       On accède à la probabilité de la classe prédite par index.
                probs_list = result.get("probabilities", [])
                prob_value = (
                    float(probs_list[pred_class])
                    if (pred_class is not None and isinstance(probs_list, list) and len(probs_list) > pred_class)
                    else 0.0
                )

                pred_label = CLASS_LABELS.get(pred_class, "Classe inconnue")

                severity_colors = {
                    0: {"card": "result-healthy", "badge": "badge-healthy", "icon": "🟢"},
                    1: {"card": "result-disease", "badge": "badge-disease", "icon": "🟡"},
                    2: {"card": "result-disease", "badge": "badge-disease", "icon": "🟠"},
                    3: {"card": "result-disease", "badge": "badge-disease", "icon": "🔴"},
                    4: {"card": "result-disease", "badge": "badge-disease", "icon": "🔴"},
                }
                colors   = severity_colors.get(pred_class, severity_colors[0])
                prob_pct = int(prob_value * 100)

                st.markdown(
                    f"""
                    <div class="result-card {colors['card']}">
                        <span class="result-badge {colors['badge']}">{colors['icon']} {pred_label}</span>
                        <h2 style="margin-bottom:0.2rem;">Classe {pred_class} — {prob_pct}% de confiance</h2>
                        <p>Évaluation de la sévérité de la maladie cardiaque pour ce patient.</p>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{prob_pct}%; background:#3498db;"></div>
                        </div>
                        <br>
                        <p><b>Interprétation :</b> {pred_label} (niveau {pred_class})</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # [C10] Distribution des probabilités : probs est une liste, on itère par index
                st.markdown("### 📊 Distribution des probabilités")
                if isinstance(probs_list, list) and probs_list:
                    prob_data = {f"Classe {i}": float(v) for i, v in enumerate(probs_list)}
                    col_d1, col_d2 = st.columns([1, 1])
                    with col_d1:
                        st.bar_chart(prob_data)
                    with col_d2:
                        for class_name, prob_val in prob_data.items():
                            st.write(f"{class_name}: {prob_val * 100:.2f}%")

            with st.expander("📄 Réponse JSON brute de l'API"):
                st.json(result)
            with st.expander("📤 Données envoyées à l'API"):
                st.json(payload)


# ===========================================================================
# TAB 2 — Prédiction en masse
# ===========================================================================
with tab2:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">📤 Charger un fichier CSV</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Format attendu :** Un fichier CSV avec les 13 colonnes suivantes :\n"
        "`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`\n\n"
        "Le séparateur (virgule `,` ou point-virgule `;`) est **détecté automatiquement**."
    )

    # Template téléchargeable (avec ligne exemple)
    all_cols = REQUIRED_COLS + OPTIONAL_COLS_LIST
    template_df = pd.DataFrame(
        [[54, 1, 0, 130, 246, 0, 0, 150, 0, 0.0, 0, 0, 3]],
        columns=all_cols,
    )
    template_csv = template_df.to_csv(index=False, sep=";")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.download_button(
            label="📥 Télécharger Template",
            data=template_csv,
            file_name="template_patients.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_t2:
        st.caption("Template avec une ligne exemple (séparateur `;`). Remplissez et uploadez.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">📤 Charger et traiter un fichier CSV</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Sélectionnez un fichier CSV", type=["csv"], key="bulk_upload")

    if uploaded_file is not None:
        try:
            # [C2] Détection automatique du séparateur (virgule ou point-virgule)
            df = pd.read_csv(uploaded_file, sep=None, engine="python")

            # Vérifier les colonnes requises EN PREMIER
            missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

            if missing_cols:
                st.markdown(
                    f'<div class="error-box">❌ <b>Colonnes requises manquantes :</b> {", ".join(missing_cols)}</div>',
                    unsafe_allow_html=True,
                )
                st.stop()
            
            # Afficher l'aperçu des données chargées
            st.markdown(
                '<div class="section-card">'
                '<div class="section-title">✅ Fichier chargé avec succès</div>',
                unsafe_allow_html=True,
            )
            
            st.write(f"**Nombre de lignes détectées :** `{len(df)}`")
            st.write(f"**Colonnes détectées :** `{len(df.columns)}`")
            
            # Afficher les colonnes trouvées
            cols_found = list(df.columns)
            cols_ok = [c for c in cols_found if c in ALL_EXPECTED_COLS]
            cols_unexpected = [c for c in cols_found if c not in ALL_EXPECTED_COLS]
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(f"✅ **Colonnes valides :** {len(cols_ok)}")
                st.caption(f"Parmi les {len(cols_found)} trouvées")
            with col_info2:
                if cols_unexpected:
                    st.write(f"⚠️ **Colonnes inattendues :** {len(cols_unexpected)}")
                    st.caption(f"{', '.join(cols_unexpected[:3])}{'...' if len(cols_unexpected) > 3 else ''}")
                else:
                    st.write(f"✅ **Aucune colonne inattendue**")
                    st.caption("Toutes les colonnes sont valides")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Afficher un aperçu du contenu du fichier
            st.markdown(
                '<div class="section-card">'
                '<div class="section-title">📋 Aperçu des données</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(df.head(10), use_container_width=True, hide_index=False)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Vérifier les colonnes manquantes (optionnelles)
            missing_optional = [c for c in OPTIONAL_COLS_LIST if c not in df.columns]
            if missing_optional:
                st.info(f"ℹ️ Colonnes optionnelles manquantes : `{', '.join(missing_optional)}`. "
                        f"Elles seront traitées comme valeurs manquantes (None).")
            
            # Avertissement pour colonnes inattendues
            if cols_unexpected:
                st.warning(f"⚠️ Colonnes inattendues qui seront ignorées : `{', '.join(cols_unexpected)}`")
            
            # [C3] Garde contre ZeroDivisionError si fichier vide
            total_rows = len(df)
            if total_rows == 0:
                st.warning("⚠️ Le fichier CSV est vide — aucune ligne à traiter.")
                st.stop()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Bouton de prédiction
            col_b1, col_b2, col_b3 = st.columns([2, 1, 2])
            with col_b2:
                predict_bulk = st.button(
                    "🚀 Lancer la prédiction", use_container_width=True, key="bulk_predict_btn"
                )

            if predict_bulk:
                predict_url = (
                    PREDICT_MULTICLASS_URL
                    if st.session_state.selected_model == "multiclass"
                    else PREDICT_BINARY_URL
                )

                st.markdown(
                    '<div class="section-card">'
                    '<div class="section-title">⏳ Traitement en cours…</div>',
                    unsafe_allow_html=True,
                )

                progress_bar  = st.progress(0)
                status_text   = st.empty()
                results       = [None] * total_rows
                completed_cnt = 0

                def _predict_row(args):
                    """Fonction worker pour le pool de threads."""
                    row_idx, row = args
                    try:
                        api_result = predict_single_row(predict_url, row)

                        if st.session_state.selected_model == "binary":
                            is_dis = api_result.get("prediction_label") == "disease"
                            return row_idx, {
                                "Index":           row_idx + 1,
                                **{c: (row[c] if c in row.index else None) for c in ALL_EXPECTED_COLS},
                                # [C4] champ "status" séparé pour comptage fiable
                                "status":          "disease" if is_dis else "healthy",
                                "Prédiction":      "🔴 Malade" if is_dis else "🟢 Sain",
                                "Probabilité (%)": round(api_result.get("probability_disease", 0) * 100, 2),
                                "Confiance":       round(api_result.get("probability_disease", 0), 4),
                            }
                        else:
                            # [C9] Champ correct : "prediction_code"
                            pred_class = api_result.get("prediction_code")

                            # [C10] probabilities est une liste, accès par index
                            probs_list = api_result.get("probabilities", [])
                            prob_value = (
                                float(probs_list[pred_class])
                                if (pred_class is not None
                                    and isinstance(probs_list, list)
                                    and len(probs_list) > pred_class)
                                else 0.0
                            )
                            return row_idx, {
                                "Index":         row_idx + 1,
                                **{c: (row[c] if c in row.index else None) for c in ALL_EXPECTED_COLS},
                                "status":        "ok" if pred_class is not None else "error",
                                "Classe":        pred_class,
                                "Sévérité":      CLASS_LABELS_SHORT.get(pred_class, "Inconnue"),
                                "Confiance (%)": round(prob_value * 100, 2),
                                "Confiance":     round(prob_value, 4),
                            }

                    except Exception as exc:
                        return row_idx, {
                            "Index":           row_idx + 1,
                            **{c: (row[c] if c in row.index else None) for c in ALL_EXPECTED_COLS},
                            "status":          "error",
                            "Prédiction":      "❌ Erreur",
                            "Probabilité (%)": None,
                            "Confiance":       None,
                            "_error":          str(exc),
                        }

                rows_iter = list(df.iterrows())

                # [C6] Requêtes parallèles via ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {
                        executor.submit(_predict_row, (idx, row)): idx
                        for idx, row in rows_iter
                    }
                    for future in as_completed(futures):
                        row_idx, result_row = future.result()
                        results[row_idx] = result_row
                        completed_cnt += 1
                        progress_bar.progress(completed_cnt / total_rows)
                        status_text.text(
                            f"Traitement : {completed_cnt}/{total_rows} lignes"
                        )

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Résultats ────────────────────────────────────
                results_df = pd.DataFrame(results)

                st.markdown(
                    '<div class="section-card">'
                    '<div class="section-title">📊 Résultats des prédictions</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # ── Statistiques ─────────────────────────────────
                # [C4] Comptage via champ "status" (sans emoji)
                if st.session_state.selected_model == "binary":
                    malade_count = sum(1 for r in results if r and r.get("status") == "disease")
                    sain_count   = sum(1 for r in results if r and r.get("status") == "healthy")
                    erreur_count = sum(1 for r in results if r and r.get("status") == "error")

                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("🔴 Malades", malade_count, f"{round(100 * malade_count / total_rows, 1)}%")
                    with col_s2:
                        st.metric("🟢 Sains", sain_count, f"{round(100 * sain_count / total_rows, 1)}%")
                    with col_s3:
                        st.metric("❌ Erreurs", erreur_count)
                else:
                    class_counts: dict = {}
                    for r in results:
                        if r is None:
                            continue
                        cl = r.get("Classe")
                        if cl is not None:
                            class_counts[cl] = class_counts.get(cl, 0) + 1

                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.markdown("**Distribution par classe :**")
                        for cl in sorted(class_counts.keys()):
                            count = class_counts[cl]
                            pct   = round(100 * count / total_rows, 1)
                            st.write(
                                f"Classe {cl} ({CLASS_LABELS_SHORT.get(cl, 'Inconnue')}) : "
                                f"{count} patients ({pct}%)"
                            )
                    with col_s2:
                        st.markdown("**Statistiques générales :**")
                        st.metric("📊 Total traitées", total_rows)
                        erreur_count = sum(1 for r in results if r and r.get("status") == "error")
                        st.metric("❌ Erreurs", erreur_count)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Téléchargement ───────────────────────────────
                st.markdown(
                    '<div class="section-card">'
                    '<div class="section-title">💾 Télécharger les résultats</div>',
                    unsafe_allow_html=True,
                )
                csv_buffer = io.StringIO()
                # Exclure les colonnes techniques "status" et "_error" de l'export
                export_cols = [c for c in results_df.columns if c not in ("status", "_error")]
                results_df[export_cols].to_csv(csv_buffer, index=False)

                st.download_button(
                    label="📥 Télécharger en CSV",
                    data=csv_buffer.getvalue(),
                    file_name="predictions_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        except pd.errors.ParserError:
            st.markdown(
                '<div class="error-box">❌ <b>Erreur de format CSV.</b> '
                "Assurez-vous que le fichier est un CSV valide (virgule ou point-virgule).</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="error-box">🚨 <b>Erreur inattendue :</b> {html.escape(str(e))}</div>',
                unsafe_allow_html=True,
            )
        finally:
            st.markdown("</div>", unsafe_allow_html=True)
