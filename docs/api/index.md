# API FastAPI

L'API expose le diagnostic par image. Elle est volontairement progressive : elle démarre même si les modèles finaux ne sont pas encore publiés. Dans ce cas, les endpoints de statut répondent normalement et les endpoints de prédiction retournent une erreur `503` explicite.

## Architecture

Le code API est séparé en plusieurs responsabilités :

| Fichier | Rôle |
|---|---|
| `src/api/main.py` | Crée l'application FastAPI et branche les routeurs |
| `src/api/preprocessing.py` | Convertit une image uploadée en batch numpy RGB brut |
| `src/api/model_loader.py` | Lit `ensemble_config.json`, charge les modèles et fait le vote doux |
| `src/api/schemas.py` | Définit les contrats Pydantic |
| `src/api/routers/health.py` | Expose le healthcheck |
| `src/api/routers/models.py` | Expose l'état de la configuration modèles |
| `src/api/routers/predict.py` | Expose les endpoints de prédiction |

Le preprocessing de l'API ne normalise pas l'image. Il redimensionne seulement l'image en RGB `224x224` et conserve des valeurs brutes `0-255`, car les modèles Keras embarquent déjà leur preprocessing.

## Configuration

Variables utiles :

| Variable | Exemple | Rôle |
|---|---|---|
| `MODEL_SOURCE` | `local` ou `hub` | Source des modèles |
| `ENSEMBLE_CONFIG_PATH` | `models/ensemble_config.json` | Chemin local optionnel vers la config |
| `HF_REPO_ID` | `<user>/plant-disease-detection-models` | Repo Hugging Face contenant les modèles |
| `HF_TOKEN` | `hf_...` | Token nécessaire si le repo HF est privé |
| `CONFIDENCE_THRESHOLD` | `0.85` | Seuil de confiance pour la détection automatique d'espèce |

En local, la valeur par défaut est `MODEL_SOURCE=local`. En production Hugging Face Spaces, utiliser `MODEL_SOURCE=hub`.

## Endpoints

### `GET /health`

Vérifie que l'API est en ligne. Cet endpoint ne charge pas TensorFlow et ne dépend pas des modèles.

Réponse :

```json
{
  "status": "ok"
}
```

Exemple :

```bash
curl http://127.0.0.1:8000/health
```

### `GET /models/info`

Retourne l'état de la configuration modèles.

Si les modèles ne sont pas encore disponibles :

```json
{
  "config_available": false,
  "source": "local",
  "complete_tasks": [],
  "missing_tasks": [],
  "loaded_model_cache_size": 0,
  "tasks": {},
  "error": "Configuration d'ensemble introuvable: ..."
}
```

Après génération de `ensemble_config.json` et publication des modèles :

```json
{
  "config_available": true,
  "source": "hub",
  "complete": true,
  "complete_tasks": ["apple", "corn", "grape", "pepper", "potato", "species", "strawberry", "tomato"],
  "missing_tasks": [],
  "loaded_model_cache_size": 0,
  "tasks": {
    "species": {
      "task_type": "species",
      "display_name": "Espèce",
      "strategy": "soft_vote_mean_probabilities",
      "class_count": 7,
      "model_count": 3,
      "architectures": ["EfficientNetB0", "ConvNeXtTiny", "EfficientNetB1"]
    }
  }
}
```

Exemple :

```bash
curl http://127.0.0.1:8000/models/info
```

### `POST /predict`

Endpoint principal. Il reçoit une image et, optionnellement, une espèce déclarée.

Si `species` est absent :

1. l'API prédit l'espèce avec l'ensemble `species` ;
2. si la confiance est inférieure à `CONFIDENCE_THRESHOLD`, elle retourne `uncertain_species` ;
3. sinon elle route vers le modèle maladie de l'espèce détectée.

Si `species` est fourni :

1. l'API considère l'espèce comme déclarée par l'utilisateur ;
2. elle route directement vers le modèle maladie correspondant.

Espèces acceptées :

```text
tomato, apple, grape, corn, potato, pepper, strawberry
```

Exemple automatique :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/samples/leaf.jpg"
```

Exemple manuel :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/samples/leaf.jpg" \
  -F "species=tomato"
```

Réponse `ok` :

```json
{
  "status": "ok",
  "species": {
    "species": "tomato",
    "confidence": 1.0,
    "source": "manual"
  },
  "disease": {
    "disease": "Late_Blight",
    "confidence": 0.91
  },
  "gradcam_base64": null,
  "action_required": null
}
```

Réponse si l'espèce automatique est incertaine :

```json
{
  "status": "uncertain_species",
  "species": {
    "species": "tomato",
    "confidence": 0.62,
    "source": "auto"
  },
  "disease": null,
  "gradcam_base64": null,
  "action_required": "Espèce détectée avec une confiance insuffisante. Merci de confirmer l'espèce avant le diagnostic."
}
```

### `POST /predict/species`

Détecte uniquement l'espèce.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/predict/species \
  -F "file=@data/samples/leaf.jpg"
```

Réponse :

```json
{
  "species": "tomato",
  "confidence": 0.96,
  "source": "auto"
}
```

### `POST /predict/disease`

Diagnostique uniquement la maladie pour une espèce fournie. L'espèce est obligatoire.

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/predict/disease \
  -F "file=@data/samples/leaf.jpg" \
  -F "species=tomato"
```

Réponse :

```json
{
  "status": "ok",
  "species": {
    "species": "tomato",
    "confidence": 1.0,
    "source": "manual"
  },
  "disease": {
    "disease": "Late_Blight",
    "confidence": 0.91
  },
  "gradcam_base64": null,
  "action_required": null
}
```

## Erreurs

| Code | Cas | Exemple de cause |
|---|---|---|
| `400` | Image invalide | Fichier vide ou non lisible par PIL |
| `422` | Entrée invalide | Espèce hors enum, champ requis manquant |
| `503` | Modèles indisponibles | `ensemble_config.json` absent, tâche absente, checkpoint manquant |

Exemple `503` avant la génération des modèles finaux :

```json
{
  "detail": "Configuration d'ensemble introuvable: ... Lance le notebook 05 pour générer models/ensemble_config.json."
}
```

## Lancement local

Sans modèles finaux, `/health` et `/models/info` permettent déjà de vérifier que l'API fonctionne.

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Documentation interactive :

```text
http://127.0.0.1:8000/docs
```

## Déploiement

En production Hugging Face Spaces, l'API utilise :

```env
MODEL_SOURCE=hub
HF_REPO_ID=<user>/plant-disease-detection-models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
CONFIDENCE_THRESHOLD=0.85
```

Le détail du déploiement est décrit dans la page [Déploiement](../deployment.md).
