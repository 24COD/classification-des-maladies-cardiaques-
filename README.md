# CardioRisk Predictor

Système de classification des maladies cardiaques utilisant le dataset UCI Heart Disease.

## Architecture

- **API** (`API/api.py`) : API FastAPI pour les prédictions individuelles et en masse
- **Interface** (`APP/app.py`) : Interface Streamlit pour l'utilisation utilisateur
- **Modèles** (`models/`) : Modèles entraînés et métadonnées

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner les modèles (si nécessaire)

```bash
python export_models.py
```

### 2. Lancer l'API

```bash
cd API
uvicorn api:app --reload --port 8000
```

L'API sera disponible sur http://localhost:8000

### 3. Lancer l'interface Streamlit

```bash
cd APP
streamlit run app.py
```

L'interface sera disponible sur http://localhost:8501

## Endpoints API

- `GET /` : Accueil avec liste des endpoints
- `GET /health` : Statut des modèles
- `POST /predict/binary` : Prédiction binaire (sain/malade)
- `POST /predict/multiclass` : Prédiction multiclasse (sévérité 0-4)
- `POST /predict/binary/batch` : Prédictions binaires en masse
- `POST /predict/multiclass/batch` : Prédictions multiclasses en masse

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