"""
app.py — Interface Streamlit pour l'API de prédiction des maladies cardiaques
=============================================================================
API cible : https://classification-des-maladies-cardiaques-1-zbq3.onrender.com
Endpoint  : POST /predict/binary | POST /predict/multiclass

FONCTIONNALITÉS :
  - Onglet 1 : Prédiction individuelle via formulaire.
  - Onglet 2 : Prédiction en masse stricte. Affiche uniquement les données 
               chargées, fait la prédiction globale, et renvoie le tableau 
               exact d'origine avec les colonnes de prédictions ajoutées.
"""

import html
import io
import os
import threading
import time
from urllib.parse import urlparse
from typing import Optional

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
# 1. CSS personnalisé (Thème Bleu et Gris)
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

API_TIMEOUT   = 60
API_MAX_RETRY = 1
API_FALLBACK_STATUS_CODES = {502, 503, 504}

REQUIRED_COLS = ["age", "sex", "cp"]
OPTIONAL_COLS_LIST = [
    "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal"
]

ALL_EXPECTED_COLS = set(REQUIRED_COLS + OPTIONAL_COLS_LIST)

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
    clean = " ".join(html.escape(text or "").split())
    return clean if len(clean) <= limit else f"{clean[:limit]}..."

def call_api(url: str, payload: dict) -> dict:
    last_exc = None
    for attempt in range(API_MAX_RETRY + 1):
        try:
            resp = requests.post(url, json=payload, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt < API_MAX_RETRY:
                time.sleep(2)
            continue
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.ConnectionError(str(e)) from e
        except requests.exceptions.HTTPError as e:
            e.response_text = _compact_response_text(resp.text)
            e.status_code   = resp.status_code
            raise e
    raise requests.exceptions.Timeout(str(last_exc))

def validate_clinical(payload: dict) -> list:
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

# ---------------------------------------------------------------------------
# 5. En-tête principal
# ---------------------------------------------------------------------------
st.markdown("<h1>Prédiction du Risque Cardiaque</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#475569; font-size:1rem; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Évaluation avancée du risque cardiovasculaire selon les données cliniques."
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
            submitted = st.form_submit_button("🔬 Lancer la prédiction", use_container_width=True)

    if submitted:
        payload = {
            "age":      float(age), "sex":      float(sex), "cp":       float(cp),
            "trestbps": float(trestbps), "chol":     float(chol), "fbs":      float(fbs),
            "restecg":  float(restecg), "thalach":  float(thalach), "exang":    float(exang),
            "oldpeak":  float(oldpeak), "slope":    float(slope), "ca":       float(ca),
            "thal":     float(thal),
        }

        clin_warnings = validate_clinical(payload)
        for w in clin_warnings:
            st.warning(f"⚠️ {w}")

        predict_url = PREDICT_MULTICLASS_URL if st.session_state.selected_model == "multiclass" else PREDICT_BINARY_URL

        with st.spinner("Analyse en cours via l'API…"):
            result = None
            try:
                result = call_api(predict_url, payload)
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">🚨 <b>Erreur :</b> {html.escape(str(e))}</div>',
                    unsafe_allow_html=True,
                )

        if result:
            if st.session_state.selected_model == "binary":
                label      = result.get("prediction_label", "healthy")
                prob       = result.get("probability_disease", 0.0)
                is_disease = (label == "disease")

                card_class  = "result-disease" if is_disease else "result-healthy"
                badge_class = "badge-disease"  if is_disease else "badge-healthy"
                icon        = "🔴" if is_disease else "🟢"
                label_fr    = "MALADIE DÉTECTÉE" if is_disease else "AUCUNE MALADIE DÉTECTÉE"
                desc        = "Risque cardiovasculaire significatif." if is_disease else "Pas de signal clinique alarmant."
                prob_pct    = int(prob * 100)
                bar_color   = "#e74c3c" if prob > 0.6 else ("#f39c12" if prob > 0.4 else "#27ae60")

                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <span class="result-badge {badge_class}">{icon} {label_fr}</span>
                        <h2 style="margin-bottom:0.2rem;">{prob_pct}% de probabilité</h2>
                        <p>{desc}</p>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{prob_pct}%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                pred_class = result.get("prediction_code", 0)
                probs_list = result.get("probabilities", [])
                prob_value = float(probs_list[pred_class]) if (isinstance(probs_list, list) and len(probs_list) > pred_class) else 0.0

                pred_label = CLASS_LABELS.get(pred_class, "Inconnue")
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
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{prob_pct}%; background:#3498db;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ===========================================================================
# TAB 2 — Prédiction en masse (Vrai Batch Processing, strict et fidèle)
# ===========================================================================
with tab2:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">📥 1. Obtenir le Template</div>',
        unsafe_allow_html=True,
    )

    template_df = pd.DataFrame([[54, 1, 0, 130, 246, 0, 0, 150, 0, 0.0, 0, 0, 3]], columns=list(ALL_EXPECTED_COLS))
    template_csv = template_df.to_csv(index=False, sep=";")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.download_button(
            label="📥 Télécharger le Template CSV",
            data=template_csv,
            file_name="template_patients.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_t2:
        st.caption("Utilisez ce modèle pour préparer vos données. Remplissez les informations, puis chargez le fichier ci-dessous.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">📤 2. Charger les données et Prédire</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Sélectionnez votre fichier complété", type=["csv"], key="bulk_upload")

    if uploaded_file is not None:
        try:
            # Lecture du fichier original sans altération
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
            
            missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing_cols:
                st.error(f"❌ Colonnes obligatoires manquantes : {', '.join(missing_cols)}")
                st.stop()
            
            if len(df) == 0:
                st.warning("⚠️ Le fichier CSV est vide.")
                st.stop()

            # Affichage strict uniquement des données chargées (intégralité du tableau)
            st.markdown("##### 📋 Données chargées prêtes pour l'analyse")
            st.dataframe(df, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns([2, 1, 2])
            with col_b2:
                predict_bulk = st.button("🚀 Lancer la prédiction en masse", use_container_width=True)

            if predict_bulk:
                predict_url = (
                    PREDICT_MULTICLASS_BATCH_CSV_URL
                    if st.session_state.selected_model == "multiclass"
                    else PREDICT_BINARY_BATCH_CSV_URL
                )

                with st.spinner("Transmission sécurisée et traitement par l'API..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": ("data.csv", uploaded_file.getvalue(), "text/csv")}
                        resp = requests.post(predict_url, files=files, timeout=120)
                        resp.raise_for_status()
                        
                        batch_data = resp.json()
                        api_results = []
                        if isinstance(batch_data, dict):
                            api_results = batch_data.get("results", batch_data.get("predictions", []))
                        elif isinstance(batch_data, list):
                            api_results = batch_data

                        # Copie exacte du dataframe original pour y ajouter uniquement les résultats
                        df_results = df.copy()
                        
                        preds, probs, classes, severities = [], [], [], []

                        # Parcours synchronisé pour correspondre chaque ligne à son résultat
                        for i in range(len(df)):
                            if i < len(api_results):
                                res = api_results[i]
                                if st.session_state.selected_model == "binary":
                                    is_disease = res.get("prediction_label") == "disease"
                                    preds.append("🔴 Malade" if is_disease else "🟢 Sain")
                                    probs.append(round(res.get("probability_disease", 0.0) * 100, 2))
                                else:
                                    pred_class = res.get("prediction_code")
                                    probs_list = res.get("probabilities", [])
                                    prob_val = float(probs_list[pred_class]) if (pred_class is not None and len(probs_list) > pred_class) else 0.0
                                    classes.append(pred_class)
                                    severities.append(CLASS_LABELS_SHORT.get(pred_class, "Inconnue"))
                                    probs.append(round(prob_val * 100, 2))
                            else:
                                # Sécurité anti-crash si la réponse de l'API est tronquée
                                preds.append("❌ Erreur")
                                probs.append(None)
                                classes.append(None)
                                severities.append("Erreur")

                        # Ajout strict des colonnes de résultats à la fin du tableau
                        if st.session_state.selected_model == "binary":
                            df_results["Prédiction"] = preds
                            df_results["Probabilité (%)"] = probs
                        else:
                            df_results["Classe Prédite"] = classes
                            df_results["Sévérité"] = severities
                            df_results["Confiance (%)"] = probs

                        st.success("✅ Traitement terminé avec succès !")

                        st.markdown("##### 📊 Résultats sur vos données")
                        st.dataframe(df_results, use_container_width=True)

                        # Bouton de téléchargement du tableau fusionné (Données originales + Prédictions)
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False, sep=";")
                        
                        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
                        with col_d2:
                            st.download_button(
                                label="💾 Télécharger les résultats complets (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name="predictions_masse_resultats.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                    except Exception as exc:
                        st.markdown(
                            f'<div class="error-box">🚨 <b>Échec du traitement :</b> {html.escape(str(exc))}</div>',
                            unsafe_allow_html=True
                        )

        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
            
        finally:
            st.markdown("</div>", unsafe_allow_html=True)