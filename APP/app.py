"""
app.py — Interface Streamlit pour l'API de prédiction des maladies cardiaques
=============================================================================
"""

import html
import io
import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

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
        --white:       #ffffff;
        --grey-100:    #f4f6f9;
        --grey-200:    #e2e8f0;
        --grey-800:    #1e293b;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--grey-100) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--grey-800) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--blue-dark) 0%, var(--blue-mid) 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #cde0f7 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--white) !important; }

    h1 { font-family: 'DM Serif Display', serif !important; color: var(--blue-dark) !important; }
    
    div.stButton > button {
        background: linear-gradient(135deg, var(--blue-mid) 0%, var(--blue-light) 100%) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(26, 74, 138, 0.35) !important;
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
    .result-disease { background: linear-gradient(135deg, #1c3a5c 0%, #2b5fa3 100%); border-left: 5px solid #e74c3c; color: var(--white) !important; }
    .result-healthy { background: linear-gradient(135deg, #0d3b6e 0%, #1a6e9a 100%); border-left: 5px solid #27ae60; color: var(--white) !important; }
    
    .result-card h2 { color: var(--white) !important; font-size: 1.7rem !important; }
    .result-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .badge-disease { background: rgba(231,76,60,0.25); color: #ff9999 !important; border: 1px solid rgba(231,76,60,0.5); }
    .badge-healthy { background: rgba(39,174,96,0.25); color: #a8f0c6 !important; border: 1px solid rgba(39,174,96,0.5); }

    .prob-bar-bg { background: rgba(255,255,255,0.15); border-radius: 8px; height: 10px; margin-top: 0.5rem; overflow: hidden; }
    .prob-bar-fill { height: 10px; border-radius: 8px; transition: width 0.5s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Configuration dynamique de l'API
# ---------------------------------------------------------------------------
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "binary"



with st.sidebar:
    st.markdown("## 🫀 CardioRisk")
    st.markdown("**Système d'aide à la décision**")
    st.markdown("---")

    st.markdown("### 🔌 Connexion API")
    # Utiliser la variable d'environnement API_BASE_URL si elle existe, sinon utiliser l'URL par défaut
    default_api_url = os.getenv("API_BASE_URL", "https://cardiorisk-api.onrender.com")
    api_url_input = st.text_input("Adresse de l'API :", value=default_api_url)
    
    API_BASE_URL = api_url_input.strip().rstrip("/")
    if not API_BASE_URL.startswith(("http://", "https://")):
        API_BASE_URL = f"https://{API_BASE_URL}"

    st.markdown("---")
    st.markdown("### ⚙️ Modèle")
    selected_model_label = st.radio(
        "Choisir le modèle :",
        options=["Binaire", "Multiclasse"],
        format_func=lambda x: f"🔵 {x}" if x == "Binaire" else f"📊 {x}"
    )
    st.session_state.selected_model = "binary" if selected_model_label == "Binaire" else "multiclass"

# Adresses de l'API (On utilise les requêtes JSON au lieu de CSV pour éviter le blocage de Render)
PREDICT_BINARY_URL           = f"{API_BASE_URL}/predict/binary"
PREDICT_MULTICLASS_URL       = f"{API_BASE_URL}/predict/multiclass"
PREDICT_BINARY_BATCH_URL     = f"{API_BASE_URL}/predict/binary/batch"
PREDICT_MULTICLASS_BATCH_URL = f"{API_BASE_URL}/predict/multiclass/batch"

REQUIRED_COLS = ["age", "sex", "cp"]
OPTIONAL_COLS_LIST = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
ALL_EXPECTED_COLS = set(REQUIRED_COLS + OPTIONAL_COLS_LIST)
CLASS_LABELS_SHORT = {0: "Pas de maladie", 1: "Légère", 2: "Modérée", 3: "Sévère", 4: "Très sévère"}

# ---------------------------------------------------------------------------
# 3. En-tête principal
# ---------------------------------------------------------------------------
st.markdown("<h1>Prédiction du Risque Cardiaque</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#475569; margin-top:-0.5rem; margin-bottom:1.5rem;'>Évaluation avancée du risque cardiovasculaire selon les données cliniques.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Prédiction Individuelle", "📊 Prédiction en Masse"])

# ===========================================================================
# TAB 1 — Prédiction individuelle
# ===========================================================================
with tab1:
    with st.form("patient_form", clear_on_submit=False):
        st.markdown('<div class="section-card"><div class="section-title">📋 Informations démographiques</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: age = st.number_input("Âge (années)", min_value=0, max_value=120, value=54, step=1)
        with col2: sex = st.selectbox("Sexe", options=[1, 0], format_func=lambda x: "Homme" if x == 1 else "Femme")
        with col3: cp = st.selectbox("Douleur thoracique (cp)", options=[0, 1, 2, 3])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">💓 Paramètres cliniques</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4: trestbps = st.number_input("Tension (mm Hg)", value=130.0)
        with col5: chol = st.number_input("Cholestérol (mg/dl)", value=246.0)
        with col6: thalach = st.number_input("Fréq. max (bpm)", value=150.0)

        col7, col8 = st.columns(2)
        with col7: fbs = st.selectbox("Glycémie > 120 (fbs)", options=[0, 1])
        with col8: exang = st.selectbox("Angine effort (exang)", options=[0, 1])
        
        col9, col10, col11 = st.columns(3)
        with col9: restecg = st.selectbox("ECG (restecg)", options=[0, 1, 2])
        with col10: slope = st.selectbox("Pente ST (slope)", options=[0, 1, 2])
        with col11: ca = st.selectbox("Vaisseaux fluo (ca)", options=[0, 1, 2, 3, 4])
        
        col12, col13 = st.columns(2)
        with col12: oldpeak = st.number_input("Dépression ST (oldpeak)", value=0.0)
        with col13: thal = st.selectbox("Thalassémie (thal)", options=[3, 6, 7])
        st.markdown("</div>", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2: submitted = st.form_submit_button("🔬 Prédiction Individuelle", use_container_width=True)

    if submitted:
        payload = {"age": float(age), "sex": float(sex), "cp": float(cp), "trestbps": float(trestbps), "chol": float(chol), "fbs": float(fbs), "restecg": float(restecg), "thalach": float(thalach), "exang": float(exang), "oldpeak": float(oldpeak), "slope": float(slope), "ca": float(ca), "thal": float(thal)}
        predict_url = PREDICT_MULTICLASS_URL if st.session_state.selected_model == "multiclass" else PREDICT_BINARY_URL

        with st.spinner("Analyse en cours via l'API…"):
            try:
                resp = requests.post(predict_url, json=payload, timeout=30)
                if resp.status_code != 200:
                    st.error(f"🚨 Erreur HTTP {resp.status_code} : {resp.text}")
                    st.stop()
                    
                result = resp.json()
                
                if st.session_state.selected_model == "binary":
                    prob = result.get("probability_disease", 0.0)
                    is_disease = (result.get("prediction_code") == 1)

                    card_class = "result-disease" if is_disease else "result-healthy"
                    badge_class = "badge-disease" if is_disease else "badge-healthy"
                    icon, label_fr = ("🔴", "MALADIE DÉTECTÉE") if is_disease else ("🟢", "AUCUNE MALADIE DÉTECTÉE")
                    prob_pct = int(prob * 100)

                    st.markdown(f"""
                        <div class="result-card {card_class}">
                            <span class="result-badge {badge_class}">{icon} {label_fr}</span>
                            <h2 style="margin-bottom:0.2rem;">{prob_pct}% de probabilité</h2>
                            <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{prob_pct}%; background:#fff;"></div></div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    pred_class = result.get("prediction_code", 0)
                    probs_list = result.get("probabilities", [])
                    prob_val = probs_list[pred_class] if len(probs_list) > pred_class else 0.0
                    
                    st.markdown(f"""
                        <div class="result-card result-disease">
                            <span class="result-badge badge-disease">🔴 Classe {pred_class}</span>
                            <h2 style="margin-bottom:0.2rem;">{int(prob_val * 100)}% de confiance</h2>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Impossible de joindre l'API : {e}")

# ===========================================================================
# TAB 2 — Prédiction en masse (NOUVELLE MÉTHODE SANS ENVOI DE FICHIER BRUT)
# ===========================================================================
with tab2:
    st.markdown('<div class="section-card"><div class="section-title">📥 1. Obtenir le Template</div>', unsafe_allow_html=True)
    template_df = pd.DataFrame([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]], columns=list(ALL_EXPECTED_COLS))
    template_csv = template_df.to_csv(index=False, sep=";")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1: st.download_button(label="📥 Télécharger le Template CSV", data=template_csv, file_name="template_patients.csv", mime="text/csv", use_container_width=True)
    with col_t2: st.caption("Utilisez ce modèle pour préparer vos données (séparateur `;`).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📤 2. Charger les données et Prédire</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Sélectionnez votre fichier complété", type=["csv"], key="bulk_upload")

    if uploaded_file is not None:
        try:
            # Lecture du fichier soumis par l'utilisateur
            raw_bytes = uploaded_file.getvalue()
            csv_text = raw_bytes.decode("utf-8-sig")
            detected_sep = ";" if ";" in csv_text.split('\n')[0] else ","
            df = pd.read_csv(io.StringIO(csv_text), sep=detected_sep)
            
            missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing_cols:
                st.error(f"❌ Colonnes obligatoires manquantes : {', '.join(missing_cols)}")
                st.stop()

            st.markdown("##### 📋 Données chargées prêtes pour l'analyse")
            st.dataframe(df, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns([2, 1, 2])
            with col_b2:
                predict_bulk = st.button("🚀 Lancer la prédiction en masse", use_container_width=True)

            if predict_bulk:
                # ICI : On cible la route JSON standard, pas la route CSV qui cause l'Erreur 405
                predict_url = PREDICT_MULTICLASS_BATCH_URL if st.session_state.selected_model == "multiclass" else PREDICT_BINARY_BATCH_URL

                with st.spinner("Formatage des données et traitement par l'API..."):
                    
                    # 1. On formate les données de l'interface vers du JSON pur
                    df_json = df.copy()
                    df_json = df_json.astype(float) # Conversion garantie
                    df_json = df_json.replace({np.nan: None}) # Le format JSON n'aime pas le "NaN"
                    patients_payload = df_json.to_dict(orient="records")
                    
                    payload = {"patients": patients_payload}
                    
                    try:
                        resp = requests.post(predict_url, json=payload, timeout=60)
                        
                        # ANTI-CRASH SILENCIEUX
                        if resp.status_code != 200:
                            st.error(f"🚨 L'API a refusé de traiter la requête (Erreur HTTP {resp.status_code}).")
                            st.info("💡 Vérifiez que l'adresse de l'API dans la barre à gauche est bien la bonne.")
                            st.code(f"URL ciblée : {predict_url}\n\nMessage du Serveur :\n{resp.text}")
                            st.stop() 
                            
                        # Si le code 200 est reçu, on extrait les prédictions
                        batch_data = resp.json()
                        api_results = batch_data.get("predictions", [])
                        
                        # Ajout des résultats aux données d'origine
                        df_results = df.copy()
                        preds, probs, classes, severities = [], [], [], []

                        for i in range(len(df)):
                            res = api_results[i]
                            if st.session_state.selected_model == "binary":
                                prob_val = res.get("probability_disease", 0.0)
                                is_disease = (res.get("prediction_code") == 1)
                                preds.append("🔴 Malade" if is_disease else "🟢 Sain")
                                probs.append(round(prob_val * 100, 2))
                            else:
                                pred_class = res.get("prediction_code", 0)
                                probs_list = res.get("probabilities", [])
                                prob_val = probs_list[pred_class] if len(probs_list) > pred_class else 0.0
                                classes.append(pred_class)
                                severities.append(CLASS_LABELS_SHORT.get(pred_class, "Inconnue"))
                                probs.append(round(prob_val * 100, 2))

                        if st.session_state.selected_model == "binary":
                            df_results["Prédiction"] = preds
                            df_results["Probabilité (%)"] = probs
                        else:
                            df_results["Classe Prédite"] = classes
                            df_results["Sévérité"] = severities
                            df_results["Confiance (%)"] = probs

                        st.success("✅ Traitement terminé avec succès !")
                        
                        malades = sum(1 for p in preds if "Malade" in str(p) or str(p) in ["1", "2", "3", "4"])
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1: st.metric("📊 Total traitées", len(df))
                        with col_s2: st.metric("🔴 Détections Positives", malades)

                        st.markdown("##### 📊 Résultats sur vos données")
                        st.dataframe(df_results, use_container_width=True)

                        # Export Final avec votre séparateur d'origine (;)
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False, sep=detected_sep)
                        
                        col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
                        with col_d2:
                            st.download_button("💾 Télécharger les résultats complets (CSV)", data=csv_buffer.getvalue(), file_name="predictions_masse_resultats.csv", mime="text/csv", use_container_width=True)

                    except requests.exceptions.ConnectionError:
                        st.error(f"🚨 Serveur API injoignable à l'adresse : {predict_url}")
                        st.stop()
                    except Exception as exc:
                        st.error(f"🚨 Échec critique du traitement : {html.escape(str(exc))}")
                        st.stop()

        except Exception as e:
            st.error(f"Erreur de lecture du fichier initial : {e}")
        finally:
            st.markdown("</div>", unsafe_allow_html=True)