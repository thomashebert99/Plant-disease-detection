# Guide de démarrage

## Prérequis

- Python `3.11.9`
- `venv` ou `pyenv`
- `pip`
- Docker, optionnel pour lancer l'API et Streamlit ensemble

## Environnement local

Le projet fixe Python via `.python-version` et utilise un virtualenv local.

```bash
pyenv install 3.11.9
pyenv local 3.11.9
make install
```

## Variantes d'installation

```bash
make install      # base locale CPU + tests + notebooks + docs
make install-gpu  # variante locale GPU TensorFlow (RTX 4090)
make test
```

## Jeux de données

- `data/processed/` contient toutes les classes retenues du projet à partir de PlantVillage.
- `data/test_ood/` contient uniquement le sous-ensemble de classes réellement disponibles dans PlantDoc après alignement.
- Une classe absente de PlantDoc n'est pas retirée du projet ni de l'entraînement : cela signifie seulement qu'elle n'est pas encore évaluée en OOD avec ce dataset.
- Si un autre dataset ou des images personnelles couvrent ces classes plus tard, elles pourront être ajoutées à l'évaluation sans changer le périmètre du modèle.

## Sans Makefile

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt -r requirements-dev.txt
```

## Conteneur API

```bash
docker compose up --build
```

Services disponibles :

- API FastAPI : `http://localhost:8000`
- documentation OpenAPI : `http://localhost:8000/docs`
- interface Streamlit : `http://localhost:8501`

Avant la génération de `models/ensemble_config.json`, les services peuvent démarrer mais les prédictions restent indisponibles.
