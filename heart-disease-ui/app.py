"""
app.py — Interface Streamlit pour l'API de prédiction des maladies cardiaques
=============================================================================
API cible : https://classification-des-maladies-cardiaques-1-zbq3.onrender.com
Endpoint  : POST /predict/binary
"""

import requests
import streamlit as st
import pandas as pd
import io

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
# 1. CSS personnalisé — Thème bleu institutionnel & gris (zéro orange)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Imports Google Fonts ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    /* ── Variables de couleur ─────────────────────────────────────── */
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

    /* ── Fond général ─────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--grey-100) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--grey-800) !important;
    }

    [data-testid="stHeader"] { background: transparent !important; }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--blue-dark) 0%, var(--blue-mid) 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #cde0f7 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--white) !important; }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* ── Titres ───────────────────────────────────────────────────── */
    h1 { font-family: 'DM Serif Display', serif !important; color: var(--blue-dark) !important; }
    h2, h3 { color: var(--blue-mid) !important; font-weight: 600 !important; }

    /* ── Bouton principal ─────────────────────────────────────────── */
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

    /* ── Inputs & Selects ─────────────────────────────────────────── */
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

    /* ── Labels des champs ────────────────────────────────────────── */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--grey-600) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* ── Cards de section ─────────────────────────────────────────── */
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

    /* ── Carte de résultat ────────────────────────────────────────── */
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

    /* ── Barre de probabilité ─────────────────────────────────────── */
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

    /* ── Bannière d'erreur ────────────────────────────────────────── */
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #721c24;
        margin-top: 1rem;
    }

    /* ── Pill info sidebar ────────────────────────────────────────── */
    .info-pill {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.82rem;
        border-left: 3px solid rgba(255,255,255,0.4);
    }

    /* ── Divider stylisé ──────────────────────────────────────────── */
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
# 2. Constante API
# ---------------------------------------------------------------------------
API_BASE_URL = "https://classification-des-maladies-cardiaques-1-zbq3.onrender.com"
PREDICT_BINARY_URL = f"{API_BASE_URL}/predict/binary"

# ---------------------------------------------------------------------------
# 3. Sidebar — Informations & contexte
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫀 CardioRisk")
    st.markdown("**Système d'aide à la décision clinique**")
    st.markdown("---")

    st.markdown("### À propos")
    st.markdown(
        "Cette interface interroge une API de Machine Learning "
        "entraînée sur le dataset **UCI Heart Disease** "
        "(303 patients, 13 variables cliniques)."
    )
    st.markdown("---")

    st.markdown("### Modèles disponibles")
    st.markdown('<div class="info-pill">🔵 <b>Binaire</b> — Sain / Malade (actif)</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-pill">⚪ <b>Multiclasse</b> — Sévérité 0–4</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Endpoint actif")
    st.code("POST /predict/binary", language="bash")

    st.markdown("---")
    st.markdown("### ⚠️ Avertissement")
    st.caption(
        "Cet outil est destiné à des fins de **démonstration et de recherche** uniquement. "
        "Il ne constitue en aucun cas un avis médical professionnel."
    )

    # Bouton de statut API
    st.markdown("---")
    if st.button("🔍 Vérifier le statut API", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=10)
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
# 4. En-tête principal
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
# Onglets pour Prédiction Individuelle vs Prédiction en Masse
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Prédiction Individuelle", "📊 Prédiction en Masse"])

# ==== TAB 1 : PRÉDICTION INDIVIDUELLE ====
with tab1:
    # ---------------------------------------------------------------------------
    # 5. Formulaire — 13 variables cliniques
    # ---------------------------------------------------------------------------
    with st.form("patient_form", clear_on_submit=False):

        # ── Section 1 : Informations démographiques ──────────────────────
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

        # ── Section 2 : Paramètres cardiovasculaires ─────────────────────
        st.markdown(
            '<div class="section-card">'
            '<div class="section-title">💓 Paramètres cardiovasculaires</div>',
            unsafe_allow_html=True,
        )
        col4, col5, col6 = st.columns(3)
        with col4:
            trestbps = st.number_input(
                "Tension artérielle au repos (mm Hg)",
                min_value=0.0, max_value=300.0, value=130.0, step=1.0,
            )
        with col5:
            chol = st.number_input(
                "Cholestérol sérique (mg/dl)",
                min_value=0.0, max_value=700.0, value=246.0, step=1.0,
            )
        with col6:
            thalach = st.number_input(
                "Fréquence cardiaque max. (bpm)",
                min_value=0.0, max_value=300.0, value=150.0, step=1.0,
            )

        col7, col8 = st.columns(2)
        with col7:
            fbs = st.selectbox(
                "Glycémie à jeun > 120 mg/dl (fbs)",
                options=[0, 1],
                format_func=lambda x: "Oui (1)" if x == 1 else "Non (0)",
            )
        with col8:
            exang = st.selectbox(
                "Angine induite par l'effort (exang)",
                options=[0, 1],
                format_func=lambda x: "Oui (1)" if x == 1 else "Non (0)",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Section 3 : Données ECG & imagerie ───────────────────────────
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
            oldpeak = st.number_input(
                "Dépression ST à l'effort (oldpeak)",
                min_value=-10.0, max_value=10.0, value=0.0, step=0.1, format="%.1f",
            )
        with col13:
            thal = st.selectbox(
                "Thalassémie (thal)",
                options=[3, 6, 7],
                format_func=lambda x: {
                    3: "3 — Normal",
                    6: "6 — Défaut fixe",
                    7: "7 — Défaut réversible",
                }[x],
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Bouton de soumission ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            submitted = st.form_submit_button("🔬 Analyser", use_container_width=True)

    # ---------------------------------------------------------------------------
    # 6. Appel API & affichage des résultats
    # ---------------------------------------------------------------------------
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

        with st.spinner("Analyse en cours via l'API…"):
            try:
                response = requests.post(
                    PREDICT_BINARY_URL,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.markdown(
                    '<div class="error-box">❌ <b>Impossible de joindre l\'API.</b> '
                    "Vérifiez que le serveur est en ligne et que l'URL est correcte.</div>",
                    unsafe_allow_html=True,
                )
                result = None
            except requests.exceptions.Timeout:
                st.markdown(
                    '<div class="error-box">⏱️ <b>Délai d\'attente dépassé.</b> '
                    "Le serveur met trop de temps à répondre (>30s). "
                    "Sur Render, le démarrage à froid peut prendre ~50s — veuillez réessayer.</div>",
                    unsafe_allow_html=True,
                )
                result = None
            except requests.exceptions.HTTPError as e:
                st.markdown(
                    f'<div class="error-box">⚠️ <b>Erreur HTTP {response.status_code}.</b> '
                    f"Détail : {response.text}</div>",
                    unsafe_allow_html=True,
                )
                result = None
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">🚨 <b>Erreur inattendue :</b> {e}</div>',
                    unsafe_allow_html=True,
                )
                result = None

        if result:
            label      = result.get("prediction_label", "N/A")
            prob       = result.get("probability_disease", 0.0)
            pred_adj   = result.get("prediction_adjusted", 0)
            thresh_adj = result.get("threshold_adjusted", 0.50)
            is_disease = (label == "disease")

            card_class  = "result-disease" if is_disease else "result-healthy"
            badge_class = "badge-disease"  if is_disease else "badge-healthy"
            icon        = "🔴" if is_disease else "🟢"
            label_fr    = "MALADIE CARDIAQUE DÉTECTÉE" if is_disease else "AUCUNE MALADIE DÉTECTÉE"
            desc        = (
                "Le modèle identifie un <b>risque cardiovasculaire significatif</b> pour ce profil patient."
                if is_disease else
                "Le modèle n'identifie <b>pas de signal clinique alarmant</b> pour ce profil patient."
            )

            # Couleur de la barre de probabilité
            prob_pct  = int(prob * 100)
            bar_color = "#e74c3c" if prob > 0.6 else ("#f39c12" if prob > 0.4 else "#27ae60")
            # Note : f39c12 est du jaune-doré (warn), mais uniquement dans la barre svg interne — pas du orange UI

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
                        <b>Seuil par défaut (0.50) :</b> {'Malade' if result.get("prediction_code") == 1 else 'Sain'} &nbsp;|&nbsp;
                        <b>Seuil ajusté ({thresh_adj}) :</b> {'Malade' if pred_adj == 1 else 'Sain'}
                        <span style="font-size:0.75rem; opacity:0.7;"> (optimisé pour le rappel)</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Détail JSON en expandeur discret
            with st.expander("📄 Réponse JSON brute de l'API"):
                st.json(result)

            # Résumé du payload envoyé
            with st.expander("📤 Données envoyées à l'API"):
                st.json(payload)

# ==== TAB 2 : PRÉDICTION EN MASSE ====
with tab2:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">📤 Charger un fichier CSV</div>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        "**Format attendu :** Un fichier CSV avec les 13 colonnes suivantes :\n"
        "`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`"
    )
    
    # Créer un template vierge à télécharger
    # Générer un template avec uniquement les en-têtes (pas de lignes vides)
    template_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    template_df = pd.DataFrame(columns=template_columns)
    template_csv = template_df.to_csv(index=False, sep=';')
    
    col_template1, col_template2 = st.columns([1, 2])
    with col_template1:
        st.download_button(
            label="📥 Télécharger Template",
            data=template_csv,
            file_name="template_patients.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_template2:
        st.caption("Téléchargez le template, remplissez vos données et uploadez-le")
    
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Sélectionnez un fichier CSV", type=["csv"], key="bulk_upload")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Lire le CSV avec le bon séparateur (point-virgule)
            df = pd.read_csv(uploaded_file, sep=';')
            
            # Afficher l'aperçu des données
            st.markdown(
                '<div class="section-card">'
                '<div class="section-title">📋 Aperçu des données</div>',
                unsafe_allow_html=True,
            )
            st.write(f"**Nombre de lignes :** {len(df)}")
            st.write(f"**Colonnes trouvées :** {list(df.columns)}")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Valider les colonnes requises
            required_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                           "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.markdown(
                    f'<div class="error-box">❌ <b>Colonnes manquantes :</b> {", ".join(missing_cols)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Bouton de lancement des prédictions
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
                with col_btn2:
                    predict_bulk = st.button("🚀 Prédire", use_container_width=True, key="bulk_predict_btn")
                
                if predict_bulk:
                    st.markdown(
                        '<div class="section-card">'
                        '<div class="section-title">⏳ Traitement en cours...</div>',
                        unsafe_allow_html=True,
                    )
                    
                    # Préparer les résultats
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_rows = len(df)
                    
                    # Appeler l'API pour chaque ligne
                    for idx, row in df.iterrows():
                        try:
                            # Construire le payload
                            payload_bulk = {
                                "age":      float(row["age"]),
                                "sex":      float(row["sex"]),
                                "cp":       float(row["cp"]),
                                "trestbps": float(row["trestbps"]),
                                "chol":     float(row["chol"]),
                                "fbs":      float(row["fbs"]),
                                "restecg":  float(row["restecg"]),
                                "thalach":  float(row["thalach"]),
                                "exang":    float(row["exang"]),
                                "oldpeak":  float(row["oldpeak"]),
                                "slope":    float(row["slope"]),
                                "ca":       float(row["ca"]),
                                "thal":     float(row["thal"]),
                            }
                            
                            # Appeler l'API
                            response = requests.post(
                                PREDICT_BINARY_URL,
                                json=payload_bulk,
                                timeout=30,
                            )
                            response.raise_for_status()
                            api_result = response.json()
                            
                            # Extraire les résultats
                            result_row = {
                                "Index": idx + 1,
                                **{col: row[col] for col in required_cols},
                                "Prédiction": "🔴 Malade" if api_result.get("prediction_label") == "disease" else "🟢 Sain",
                                "Probabilité (%)": round(api_result.get("probability_disease", 0) * 100, 2),
                                "Confiance": round(api_result.get("probability_disease", 0), 4),
                            }
                            results.append(result_row)
                            
                        except Exception as e:
                            # En cas d'erreur, enregistrer l'erreur
                            result_row = {
                                "Index": idx + 1,
                                **{col: row[col] for col in required_cols},
                                "Prédiction": "❌ Erreur",
                                "Probabilité (%)": None,
                                "Confiance": None,
                            }
                            results.append(result_row)
                        
                        # Mettre à jour la barre de progression
                        progress = (idx + 1) / total_rows
                        progress_bar.progress(progress)
                        status_text.text(f"Traitement : {idx + 1}/{total_rows} lignes")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Afficher les résultats
                    st.markdown(
                        '<div class="section-card">'
                        '<div class="section-title">📊 Résultats des prédictions</div>',
                        unsafe_allow_html=True,
                    )
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Statistiques
                    malade_count = sum(1 for r in results if "Malade" in r["Prédiction"])
                    sain_count = sum(1 for r in results if "Sain" in r["Prédiction"])
                    erreur_count = sum(1 for r in results if "Erreur" in r["Prédiction"])
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("🔴 Malades", malade_count, f"{round(100*malade_count/total_rows, 1)}%")
                    with col_stat2:
                        st.metric("🟢 Sains", sain_count, f"{round(100*sain_count/total_rows, 1)}%")
                    with col_stat3:
                        st.metric("❌ Erreurs", erreur_count)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Téléchargement CSV
                    st.markdown(
                        '<div class="section-card">'
                        '<div class="section-title">💾 Télécharger les résultats</div>',
                        unsafe_allow_html=True,
                    )
                    
                    # Préparer le fichier CSV pour le téléchargement
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Télécharger en CSV",
                        data=csv_data,
                        file_name="predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
        except pd.errors.ParserError:
            st.markdown(
                '<div class="error-box">❌ <b>Erreur de format CSV.</b> '
                "Assurez-vous que le fichier est un CSV valide.</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="error-box">🚨 <b>Erreur inattendue :</b> {str(e)}</div>',
                unsafe_allow_html=True,
            )
