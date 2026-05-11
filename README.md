# 🫀 CardioRisk Predictor

**Système intelligent de classification et prédiction des maladies cardiaques**

Projet complet de Machine Learning pour l'analyse et la prédiction du risque cardiaque basé sur le dataset UCI Heart Disease. Le projet inclut une API FastAPI performante et une interface web Streamlit conviviale.

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Utilisation](#utilisation)
- [Endpoints API](#endpoints-api)
- [Déploiement](#déploiement)
- [Notebook d'analyse](#notebook-danalyse)

---

## 🎯 Aperçu du projet

### Objectif
Ce projet propose deux modèles de Machine Learning pour prédire les maladies cardiaques :
- **Modèle binaire** : Détermine si une personne est saine ou malade
- **Modèle multiclasse** : Évalue le niveau de sévérité (0 = sain, 1-4 = niveaux de maladie)

### Dataset
- **Source** : UCI Heart Disease Dataset
- **Localisation** : `/heart+disease/` (fichiers de données brutes)
- **Caractéristiques** : 13 variables cliniques et démographiques

### Technologie
- **Machine Learning** : scikit-learn
- **API** : FastAPI (uvicorn)
- **Interface** : Streamlit
- **Déploiement** : Render (via Blueprint)
- **Python** : 3.11.9+

---

## 📁 Structure du projet

```
classification maladies cardiaques/
├── README.md                          # Ce fichier
├── DEPLOYMENT.md                      # Guide de déploiement sur Render
├── requirements.txt                   # Dépendances du projet
├── runtime.txt                        # Version Python pour Render
├── render.yaml                        # Configuration Blueprint Render
├── test_startup.py                    # Script de test au démarrage
│
├── API/                               # API FastAPI
│   ├── api.py                        # Points d'accès (endpoints)
│   ├── __init__.py
│   └── requirements.txt               # Dépendances spécifiques API
│
├── APP/                               # Interface Streamlit
│   ├── app.py                        # Application web interactive
│   └── __init__.py
│
├── notebooks/                         # Analyse et modélisation
│   ├── Prediction_maladie_cardiaque_EDA.ipynb          # Exploration des données
│   └── Prediction_maladie_cardiaque_Modelisation.ipynb # Entraînement modèles
│
└── heart+disease/                     # Données brutes UCI
    ├── heart-disease.names            # Description des variables
    ├── cleve.mod                      # Données Cleveland
    ├── costs/                         # Coûts associés aux erreurs
    └── ...
```

---

## ⚙️ Installation

### Prérequis
- Python 3.11+ installé
- pip ou conda
- Git

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone <URL_REPOSITORY>
cd "classification maladies cardiaques"
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

---

## 🚀 Démarrage rapide

### Option 1 : Lancer l'interface web (Streamlit) - Recommandé pour débuter
```bash
cd APP
streamlit run app.py
```
L'interface sera accessible à : **http://localhost:8501**

### Option 2 : Lancer l'API seule
```bash
cd API
uvicorn api:app --reload --port 8000
```
L'API sera accessible à : **http://localhost:8000**
- Documentation interactive : http://localhost:8000/docs
- Vérifier l'état : http://localhost:8000/health

### Option 3 : Lancer les deux services simultanément
Ouvrir deux terminaux différents et exécuter les commandes des Options 1 et 2.

---

## 📊 Utilisation

### Via l'interface Streamlit (APP)
1. Accédez à http://localhost:8501
2. Remplissez le formulaire avec les données cliniques du patient
3. Sélectionnez le type de prédiction :
   - **Binaire** : Sain / Malade
   - **Multiclasse** : Niveau de sévérité (0-4)
4. Consultez les résultats et la confiance du modèle

### Via l'API FastAPI (Développeurs)

#### Prédiction unique (binaire)
```bash
curl -X POST "http://localhost:8000/predict/binary" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 1,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }'
```

#### Prédictions en masse (CSV)
```bash
curl -X POST "http://localhost:8000/predict/binary/batch" \
  -F "file=@patients.csv"
```

---

## 🔌 Endpoints API

### Informations générales
- `GET /` - Accueil avec liste des endpoints disponibles
- `GET /health` - Vérifier l'état des modèles et des dépendances

### Prédictions individuelles
- `POST /predict/binary` - Prédiction binaire (sain/malade) pour un patient
- `POST /predict/multiclass` - Prédiction multiclasse (sévérité 0-4) pour un patient

### Prédictions en masse
- `POST /predict/binary/batch` - Prédictions binaires depuis un fichier CSV
- `POST /predict/multiclass/batch` - Prédictions multiclasses depuis un fichier CSV

**Documentation complète** : http://localhost:8000/docs (Swagger UI)

---

## 🌐 Déploiement

### Déploiement sur Render (recommandé)

Voir le fichier [DEPLOYMENT.md](DEPLOYMENT.md) pour des instructions détaillées.

**Déploiement via Blueprint** (automatisé) :
1. Allez sur https://render.com
2. Créez un nouveau service via **New +** → **Blueprint**
3. Connectez votre repository GitHub
4. Render lira automatiquement `render.yaml` et créera deux services :
   - `cardiorisk-api` (FastAPI)
   - `cardiorisk-streamlit` (Streamlit)

**Points d'accès après déploiement** :
- API : `https://votre-api-url.onrender.com/health`
- Documentation API : `https://votre-api-url.onrender.com/docs`
- Interface : `https://votre-interface-url.onrender.com`

---

## 📓 Notebook d'analyse

Deux notebooks Jupyter sont disponibles dans `/notebooks/` pour comprendre le projet :

1. **EDA.ipynb** - Exploratory Data Analysis
   - Analyse des données du UCI Heart Disease
   - Statistiques descriptives
   - Visualisations des relations entre variables

2. **Modelisation.ipynb** - Modélisation et entraînement
   - Préparation des données
   - Entraînement des deux modèles
   - Évaluation des performances
   - Métriques et courbes ROC

### Lancer les notebooks
```bash
jupyter notebook notebooks/
```

---

## 🔧 Fichiers de configuration importants

| Fichier | Description |
|---------|-----------|
| `requirements.txt` | Dépendances pip principales du projet |
| `render.yaml` | Configuration Render pour le déploiement automatisé |
| `runtime.txt` | Spécifie Python 3.11.9 (important pour Render) |
| `.python-version` | Version locale de Python pour pyenv/direnv |
| `test_startup.py` | Script vérifiant que tous les modèles sont chargés correctement |

---

## 📞 Support et questions

Pour des questions ou des problèmes :
1. Consultez les logs dans le terminal
2. Vérifiez que Python 3.11+ est installé : `python --version`
3. Vérifiez que le dossier `models/` contient les fichiers `.joblib`
4. Consultez [DEPLOYMENT.md](DEPLOYMENT.md) pour l'aide au déploiement

---

## 📝 Licence

Ce projet est créé à des fins éducatives en Machine Learning.

## Documentation

- API interactive : http://localhost:8000/docs
- Documentation ReDoc : http://localhost:8000/redoc

## Métriques des modèles

- **Modèle binaire** : ROC-AUC = 0.9535
- **Modèle multiclasse** : ROC-AUC = 0.7694
- **Seuil ajusté** : 0.1 (optimisé pour le recall)

## Variables cliniques

13 variables d'entrée :
- `age` : Âge en années
- `sex` : Sexe (1=homme, 0=femme)
- `cp` : Type de douleur thoracique (0-3)
- `trestbps` : Tension artérielle au repos (mm Hg)
- `chol` : Cholestérol sérique (mg/dl)
- `fbs` : Glycémie à jeun > 120 mg/dl (1=oui, 0=non)
- `restecg` : Résultats ECG au repos (0-2)
- `thalach` : Fréquence cardiaque maximale
- `exang` : Angine induite par l'effort (1=oui, 0=non)
- `oldpeak` : Dépression ST à l'effort
- `slope` : Pente du segment ST (0-2)
- `ca` : Nombre de vaisseaux colorés (0-3)
- `thal` : Thalassémie (3=normal, 6=défaut fixe, 7=défaut réversible)