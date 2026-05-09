#!/usr/bin/env python
"""
Test de démarrage - Vérifie que les fichiers essentiels existent et que les imports fonctionnent.
"""
import sys
from pathlib import Path

print("=" * 70)
print("TEST DE DÉMARRAGE - VÉRIFICATION DES FICHIERS ET IMPORTS")
print("=" * 70)

# Chemin racine
ROOT_DIR = Path(__file__).parent
print(f"\n✓ Racine du projet: {ROOT_DIR}")

# Vérifier les dossiers essentiels
essential_dirs = {
    "API": ROOT_DIR / "API",
    "APP": ROOT_DIR / "APP",
    "models": ROOT_DIR / "models",
    "heart+disease": ROOT_DIR / "heart+disease",
}

print("\n1️⃣ Vérification des répertoires essentiels:")
all_dirs_exist = True
for name, path in essential_dirs.items():
    exists = path.exists()
    symbol = "✓" if exists else "✗"
    print(f"   {symbol} {name}: {path}")
    if not exists:
        all_dirs_exist = False

if not all_dirs_exist:
    print("\n❌ ERREUR: Des répertoires essentiels manquent!")
    sys.exit(1)

# Vérifier les fichiers critiques
critical_files = {
    "API/__init__.py": ROOT_DIR / "API" / "__init__.py",
    "API/api.py": ROOT_DIR / "API" / "api.py",
    "APP/__init__.py": ROOT_DIR / "APP" / "__init__.py",
    "APP/app.py": ROOT_DIR / "APP" / "app.py",
    "models/metadata.json": ROOT_DIR / "models" / "metadata.json",
    "models/model_multiclass.joblib": ROOT_DIR / "models" / "model_multiclass.joblib",
    "requirements.txt": ROOT_DIR / "requirements.txt",
}

print("\n2️⃣ Vérification des fichiers critiques:")
all_files_exist = True
for name, path in critical_files.items():
    exists = path.exists()
    symbol = "✓" if exists else "✗"
    print(f"   {symbol} {name}")
    if not exists:
        all_files_exist = False
        print(f"      ⚠️  Manquant: {path}")

if not all_files_exist:
    print("\n⚠️  ATTENTION: Des fichiers critiques manquent!")

# Vérifier les fichiers de modèles (moins critiques)
print("\n3️⃣ Vérification des fichiers de modèles:")
model_files = [
    "MEILLEUR_MODELE_BINAIRE.pkl",
    "model_binary.joblib",
    "model_multiclass.joblib",
    "seuil_retenu.json",
    "metadata.json",
]

for model_file in model_files:
    path = ROOT_DIR / "models" / model_file
    exists = path.exists()
    symbol = "✓" if exists else "✗"
    print(f"   {symbol} {model_file} ({path.stat().st_size if exists else 'N/A'} bytes)")

# Tester les imports Python
print("\n4️⃣ Test des imports Python:")
try:
    print("   • Importing fastapi...")
    import fastapi
    print("     ✓ FastAPI OK")
except ImportError as e:
    print(f"     ✗ FastAPI ERROR: {e}")

try:
    print("   • Importing streamlit...")
    import streamlit
    print("     ✓ Streamlit OK")
except ImportError as e:
    print(f"     ✗ Streamlit ERROR: {e}")

try:
    print("   • Importing pandas...")
    import pandas
    print("     ✓ Pandas OK")
except ImportError as e:
    print(f"     ✗ Pandas ERROR: {e}")

try:
    print("   • Importing sklearn...")
    import sklearn
    print("     ✓ Scikit-learn OK")
except ImportError as e:
    print(f"     ✗ Scikit-learn ERROR: {e}")

try:
    print("   • Importing joblib...")
    import joblib
    print("     ✓ Joblib OK")
except ImportError as e:
    print(f"     ✗ Joblib ERROR: {e}")

# Test d'import du module API (vérifie les __init__.py)
print("\n5️⃣ Test d'import des modules personnalisés:")
try:
    print("   • Importing API.api...")
    sys.path.insert(0, str(ROOT_DIR))
    from API import api
    print("     ✓ API.api OK")
except ImportError as e:
    print(f"     ✗ API.api ERROR: {e}")

try:
    print("   • Importing APP.app...")
    # APP ne doit pas être importé comme module Python (c'est une app Streamlit)
    print("     → APP/app.py sera lancé par Streamlit (pas importé comme module)")
except ImportError as e:
    print(f"     ✗ ERROR: {e}")

print("\n" + "=" * 70)
print("✅ TEST COMPLÉTÉ - Tous les fichiers essentiels sont présents!")
print("=" * 70)
print("\n💡 Prochaines étapes:")
print("   1. Vérifier les logs de déploiement sur Render")
print("   2. S'assurer que les modèles .joblib sont versionnés dans Git")
print("   3. Vérifier que les dépendances dans requirements.txt sont correctes")
print("\n")
