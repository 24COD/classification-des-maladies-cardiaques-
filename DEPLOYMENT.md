# Deploiement Render

## Methode recommandee : Blueprint

Le fichier `render.yaml` a la racine peut creer automatiquement les deux
services :

- `cardiorisk-api`
- `cardiorisk-streamlit`

Dans Render, choisissez **New +** puis **Blueprint**, connectez le repo GitHub,
et selectionnez ce repository. Render lira `render.yaml`.

Le projet force Python `3.11.9` avec `.python-version` et `PYTHON_VERSION`
dans `render.yaml`. C'est important parce que les versions recentes de Render
utilisent Python 3.14 par defaut, ce qui peut forcer `pandas` a compiler depuis
les sources.

## API FastAPI

Deployez le projet depuis la racine du repository pour que le dossier `models/`
soit disponible.

Build command :

```bash
pip install -r requirements.txt
```

Start command :

```bash
uvicorn API.api:app --host 0.0.0.0 --port $PORT
```

Apres le deploiement, verifiez :

```text
https://URL-DE-TON-API-RENDER/health
https://URL-DE-TON-API-RENDER/docs
```

## Interface Streamlit

Build command :

```bash
pip install -r requirements.txt
```

Start command :

```bash
streamlit run APP/app.py --server.address 0.0.0.0 --server.port $PORT
```

Variable d'environnement a definir dans le service Streamlit :

```bash
API_BASE_URL=https://URL-DE-TON-API-RENDER
```

Exemple : si l'API est disponible sur `https://cardiorisk-api.onrender.com`,
mettez exactement cette URL, sans `/predict/binary` a la fin.

Sur le plan gratuit Render, utilisez l'URL publique `.onrender.com` pour
`API_BASE_URL`. L'adresse interne `hostport` n'est pas adaptee ici, car les
Web Services gratuits ne recoivent pas de trafic prive entrant.
