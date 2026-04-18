# Déploiement et Docker

Cette page décrit le lancement local avec Docker Compose et le déploiement de l'API FastAPI sur Hugging Face Spaces.

## Services utilisés

Le projet utilise deux ressources Hugging Face :

| Ressource | Rôle |
|---|---|
| Model repo `HF_REPO_ID` | Stockage des checkpoints `.keras` et de `ensemble_config.json` |
| Space Docker API | Hébergement public de l'API FastAPI |

## Variables à configurer dans le Space API

Dans le Space API, ouvrir `Settings` puis `Variables and secrets`.

Ajouter ces valeurs :

```env
MODEL_SOURCE=hub
HF_REPO_ID=<username>/plant-disease-detection-models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
CONFIDENCE_THRESHOLD=0.85
```

`HF_TOKEN` doit avoir un droit de lecture sur le repo modèle si ce repo est privé.

## Docker

Le `Dockerfile` du projet est compatible Hugging Face Spaces :

- l'API écoute sur le port `7860` par défaut ;
- `HF_HOME=/tmp/huggingface` place le cache Hugging Face dans un dossier inscriptible ;
- `MODEL_SOURCE=hub` indique à l'API de télécharger la config et les modèles depuis le Hub.

En local, on peut toujours surcharger le port :

```bash
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 -e PORT=8000 plant-disease-api
```

## Développement local avec Docker Compose

Le fichier `docker-compose.yml` lance deux services :

| Service | Rôle | URL locale |
|---|---|---|
| `api` | FastAPI, chargement des modèles et prédiction | `http://localhost:8000` |
| `streamlit` | Interface utilisateur | `http://localhost:8501` |

Lancer les deux services :

```bash
docker compose up --build
```

Dans Compose, l'API utilise :

```env
MODEL_SOURCE=local
ENSEMBLE_CONFIG_PATH=/app/models/ensemble_config.json
```

Le dossier local `models/` est monté dans le conteneur API en lecture seule. Cela évite de copier les checkpoints dans l'image Docker.

Avant la fin du notebook 05, il est normal que `/models/info` indique que la configuration n'est pas disponible. L'API et Streamlit peuvent quand même démarrer ; seules les prédictions retournent une erreur claire tant que `models/ensemble_config.json` n'existe pas.

Après la fin des notebooks 04 puis 05 :

1. vérifier que `models/ensemble_config.json` existe ;
2. relancer `docker compose up --build` si les conteneurs ne tournent pas déjà ;
3. ouvrir `http://localhost:8000/models/info` ;
4. ouvrir `http://localhost:8501` pour tester l'interface.

Streamlit appelle l'API avec cette variable interne au réseau Docker :

```env
API_URL=http://api:7860
```

## Publication des modèles

Après la fin du notebook 04 puis du notebook 05, le fichier suivant doit exister :

```text
models/ensemble_config.json
```

Faire d'abord une simulation :

```bash
python scripts/push_models_to_hub.py --dry-run
```

Puis uploader réellement :

```bash
python scripts/push_models_to_hub.py
```

Le script lit `HF_REPO_ID` et `HF_TOKEN` depuis `.env`.

## Vérifications API

Quand le Space API est construit, vérifier :

```text
https://<username>-plant-disease-detection-api.hf.space/health
https://<username>-plant-disease-detection-api.hf.space/models/info
```

Résultat attendu :

- `/health` retourne `{"status": "ok"}` ;
- `/models/info` retourne `config_available: true` après upload des modèles.

Si `config_available` vaut `false`, vérifier :

- que `MODEL_SOURCE=hub` est bien défini dans le Space ;
- que `HF_REPO_ID` pointe vers le repo modèle ;
- que `HF_TOKEN` a accès au repo si celui-ci est privé ;
- que `scripts/push_models_to_hub.py` a bien uploadé `ensemble_config.json`.
