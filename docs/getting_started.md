# Guide de démarrage

## Prérequis

- Python `3.11.9`
- `pyenv` recommandé ;
- `pip`
- Docker, optionnel pour lancer l'API et Streamlit ensemble

## Environnement local

Le projet fixe Python via `.python-version`. L'environnement principal peut être un environnement `pyenv`, ce qui évite de dépendre d'un dossier `.venv` dans le projet.

```bash
pyenv install 3.11.9
pyenv local 3.11.9
make install
```

Le Makefile sait utiliser l'interpréteur Python actif si le virtualenv local `.venv-plant-disease-detection` n'existe pas.

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
pyenv local 3.11.9
pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt -r requirements-dev.txt
```

Une variante avec `venv` reste possible, mais elle n'est pas obligatoire pour ce projet.

## Conteneur API

```bash
docker compose up --build
```

Services disponibles :

- API FastAPI : `http://localhost:8000`
- documentation OpenAPI : `http://localhost:8000/docs`
- interface Streamlit : `http://localhost:8501`

Avant la génération de `models/ensemble_config.json`, les services peuvent démarrer mais les prédictions restent indisponibles.

## Démo en ligne

Une fois le projet publié, les services publics sont :

| Service | URL |
|---|---|
| Interface Streamlit | `https://dredfury-plant-disease-detection-app.hf.space` |
| API FastAPI | `https://dredfury-plant-disease-detection-api.hf.space` |

Pour un test rapide :

```bash
curl https://dredfury-plant-disease-detection-api.hf.space/health
curl https://dredfury-plant-disease-detection-api.hf.space/monitoring/summary
```
