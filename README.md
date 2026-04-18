# Plant Disease Detection

Application de diagnostic foliaire par image. Le projet entraîne plusieurs modèles de classification, sélectionne un ensemble final, expose les prédictions via une API FastAPI et fournit une interface Streamlit simple pour tester le diagnostic.

## Objectif

Le système répond à deux questions :

1. quelle espèce végétale est visible sur l'image ;
2. quelle maladie est détectée pour cette espèce, si les modèles sont disponibles.

L'approche retenue sépare volontairement le projet en trois blocs :

- notebooks d'analyse, d'entraînement et de sélection des modèles ;
- API FastAPI responsable du chargement des modèles et de l'inférence ;
- interface Streamlit qui appelle l'API sans charger TensorFlow.

## Architecture

```text
Image utilisateur
  -> Streamlit
  -> API FastAPI
  -> modèle espèce
  -> modèle maladie spécialisé
  -> réponse JSON + affichage utilisateur
```

Les modèles finaux sont décrits par `models/ensemble_config.json`, généré par le notebook 05 après la fin des benchmarks. En production, cette configuration et les checkpoints sont publiés sur Hugging Face Hub, puis chargés par l'API déployée sur Hugging Face Spaces.

## Installation locale

Prérequis :

- Python 3.11.9 ;
- `pyenv` recommandé ;
- Docker optionnel pour le lancement API + interface.

Installation CPU :

```bash
pyenv install 3.11.9
pyenv local 3.11.9
make install
```

Installation GPU locale :

```bash
make install-gpu
make verify-gpu
```

## Commandes utiles

```bash
make test        # lancer les tests
make docs        # servir la documentation MkDocs
make run-api     # lancer FastAPI sur http://localhost:8000
make run-app     # lancer Streamlit sur http://localhost:8501
make push-models # publier les modèles finaux sur Hugging Face Hub
```

Lancement complet avec Docker Compose :

```bash
docker compose up --build
```

URLs locales :

- API : `http://localhost:8000`
- documentation API : `http://localhost:8000/docs`
- interface Streamlit : `http://localhost:8501`

Tant que `models/ensemble_config.json` n'existe pas, l'API et Streamlit peuvent démarrer, mais les prédictions retournent une erreur claire indiquant que les modèles ne sont pas encore disponibles.

## Organisation des notebooks

| Notebook | Rôle |
|---|---|
| `01_data_exploration.ipynb` | exploration du dataset |
| `02_preprocessing.ipynb` | préparation et validation des transformations |
| `03_benchmark_species.ipynb` | benchmark des modèles de reconnaissance d'espèce |
| `04_benchmark_diseases.ipynb` | benchmark des modèles maladie par espèce |
| `05_ensemble_selection.ipynb` | sélection finale et génération de `ensemble_config.json` |
| `06_gradcam.ipynb` | interprétabilité visuelle, optionnelle pour l'API |

Le notebook 05 doit être lancé seulement après la fin complète des benchmarks du notebook 04.

## API et interface

Endpoints principaux :

- `GET /health` : état du service ;
- `GET /models/info` : disponibilité de la configuration modèle ;
- `POST /predict` : prédiction complète ;
- `POST /predict/species` : prédiction espèce seule ;
- `POST /predict/disease` : prédiction maladie avec espèce fournie.

L'interface Streamlit reste volontairement légère : elle envoie l'image à l'API, affiche les résultats et signale clairement si les modèles ne sont pas encore chargés.

## Déploiement prévu

Le projet utilise Hugging Face pour garder un déploiement simple et gratuit :

- Hugging Face Hub pour stocker `ensemble_config.json` et les checkpoints `.keras` ;
- Hugging Face Space Docker pour l'API FastAPI ;
- Hugging Face Space pour l'interface Streamlit.

La procédure détaillée se trouve dans la documentation :

```bash
make docs
```

Puis ouvrir la page `Déploiement`.

## Vérifications

```bash
make test
python -m mkdocs build --strict
docker compose config
```

Ces commandes vérifient respectivement les tests Python, la documentation et la configuration Docker Compose.
