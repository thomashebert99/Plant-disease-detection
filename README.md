# Plant Disease Detection

Application de diagnostic foliaire par image. Le projet entraîne plusieurs modèles de classification, sélectionne un ensemble final, expose les prédictions via une API FastAPI et fournit une interface Streamlit simple pour tester le diagnostic.

## Démo en ligne

| Service | URL |
|---|---|
| Interface Streamlit | `https://dredfury-plant-disease-detection-app.hf.space` |
| API FastAPI | `https://dredfury-plant-disease-detection-api.hf.space` |
| Documentation API | `https://dredfury-plant-disease-detection-api.hf.space/docs` |
| Modèles Hugging Face | `https://huggingface.co/DredFury/plant-disease-detection-models` |

Endpoints rapides :

```text
https://dredfury-plant-disease-detection-api.hf.space/health
https://dredfury-plant-disease-detection-api.hf.space/models/info
https://dredfury-plant-disease-detection-api.hf.space/monitoring/summary
```

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

## Résultats synthétiques

Le notebook 05 retient trois modèles par tâche avec la stratégie `top3_max2_family`, puis l'API utilise un vote doux par moyenne des probabilités.

| Élément | Résultat |
|---|---|
| Tâches couvertes | `species`, `tomato`, `apple`, `grape`, `corn`, `potato`, `pepper`, `strawberry` |
| Modèles finaux | 24 checkpoints Keras, 3 par tâche |
| Meilleur gain test | `tomato`, +0.0040 F1 macro avec le vote doux |
| Test in-distribution | F1 macro proche de 1.0 sur la majorité des tâches PlantVillage |
| Test OOD | performances nettement plus faibles sur PlantDoc, ce qui confirme un écart domaine contrôlé/terrain |

Les détails complets sont dans la page `Résultats` de la documentation.

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
- `GET /monitoring/summary` : synthèse des prédictions et erreurs API ;
- `POST /predict` : prédiction complète ;
- `POST /predict/species` : prédiction espèce seule ;
- `POST /predict/disease` : prédiction maladie avec espèce fournie.

L'interface Streamlit reste volontairement légère : elle envoie l'image à l'API, affiche les résultats et signale clairement si les modèles ne sont pas encore chargés. Elle affiche aussi les trois classes les plus probables pour l'espèce et la maladie, afin de mieux comprendre les hésitations du modèle. Une page `Monitoring` séparée permet de consulter une synthèse des prédictions enregistrées par l'API.

Le monitoring de production reste volontairement minimal : l'API écrit un événement JSONL par prédiction, sans stocker l'image uploadée. Ce choix permet de démontrer l'observabilité du service tout en respectant les contraintes de temps, de coût et de simplicité d'un projet réalisé seul.

## Déploiement

Le projet utilise Hugging Face pour garder un déploiement simple et gratuit :

- Hugging Face Hub pour stocker `ensemble_config.json` et les checkpoints `.keras` ;
- Hugging Face Space Docker pour l'API FastAPI ;
- Hugging Face Space pour l'interface Streamlit.

En production, l'API utilise `CONFIDENCE_THRESHOLD=0.65` pour décider si l'espèce détectée automatiquement est assez fiable avant de lancer le diagnostic maladie.

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
