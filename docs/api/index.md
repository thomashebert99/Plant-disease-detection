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
| `HF_REPO_ID` | `DredFury/plant-disease-detection-models` | Repo Hugging Face contenant les modèles |
| `HF_TOKEN` | `hf_...` | Token nécessaire si le repo HF est privé |
| `CONFIDENCE_THRESHOLD` | `0.65` | Seuil de confiance pour la détection automatique d'espèce |
| `MONITORING_LOG_PATH` | `logs/predictions.jsonl` | Fichier JSONL de monitoring des prédictions |

En local, la valeur par défaut est `MODEL_SOURCE=local`. En production Hugging Face Spaces, utiliser `MODEL_SOURCE=hub`.

URL publique de l'API :

```text
https://dredfury-plant-disease-detection-api.hf.space
```

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

Exemple production :

```bash
curl https://dredfury-plant-disease-detection-api.hf.space/health
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

Exemple production :

```bash
curl https://dredfury-plant-disease-detection-api.hf.space/models/info
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
    "source": "manual",
    "top_predictions": []
  },
  "disease": {
    "disease": "Late_Blight",
    "confidence": 0.91,
    "top_predictions": [
      {"label": "Late_Blight", "confidence": 0.91},
      {"label": "Early_Blight", "confidence": 0.05},
      {"label": "Healthy", "confidence": 0.02}
    ]
  },
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
    "source": "auto",
    "top_predictions": [
      {"label": "tomato", "confidence": 0.62},
      {"label": "pepper", "confidence": 0.21},
      {"label": "potato", "confidence": 0.08}
    ]
  },
  "disease": null,
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
  "source": "auto",
  "top_predictions": [
    {"label": "tomato", "confidence": 0.96},
    {"label": "pepper", "confidence": 0.02},
    {"label": "potato", "confidence": 0.01}
  ]
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
    "source": "manual",
    "top_predictions": []
  },
  "disease": {
    "disease": "Late_Blight",
    "confidence": 0.91,
    "top_predictions": [
      {"label": "Late_Blight", "confidence": 0.91},
      {"label": "Early_Blight", "confidence": 0.05},
      {"label": "Healthy", "confidence": 0.02}
    ]
  },
  "action_required": null
}
```

### `GET /monitoring/summary`

Retourne une synthèse des prédictions traitées par l'API, avec alertes, feedback et signaux de drift.

L'API écrit un événement JSONL par appel de prédiction, sans stocker l'image uploadée. Les informations suivies sont volontairement limitées :

- endpoint appelé ;
- mode automatique ou manuel ;
- statut `ok`, `uncertain_species` ou `error` ;
- espèce et maladie prédites quand disponibles ;
- confiances ;
- temps de réponse ;
- source des modèles, locale ou Hugging Face Hub ;
- métriques image dérivées : luminosité, contraste, netteté, saturation, ratio vert/brun.

Exemple :

```bash
curl http://127.0.0.1:8000/monitoring/summary
```

Exemple production :

```bash
curl https://dredfury-plant-disease-detection-api.hf.space/monitoring/summary
```

Réponse :

```json
{
  "enabled": true,
  "storage": "jsonl",
  "total_events": 12,
  "total_predictions": 12,
  "ok": 9,
  "uncertain_species": 2,
  "errors": 1,
  "error_rate": 0.0833,
  "uncertain_rate": 0.1667,
  "low_confidence_rate": 0.25,
  "average_latency_ms": 842.41,
  "p95_latency_ms": 1320.5,
  "average_species_confidence": 0.81,
  "average_disease_confidence": 0.76,
  "species_distribution": {"tomato": 5, "apple": 2},
  "disease_distribution": {"Late_Blight": 3, "Healthy": 2},
  "alerts": [],
  "domain_shift": {
    "status": "ood_like",
    "risk_level": "watch",
    "closest_reference": "plantdoc_ood"
  },
  "model_quality_shift": {
    "status": "insufficient_feedback",
    "risk_level": "none",
    "disagreement_rate": 0.0
  },
  "last_event_at": "2026-04-18T10:00:00+00:00"
}
```

### `GET /monitoring/events`

Retourne les derniers événements de prédiction, sans image brute, pour alimenter les graphes de monitoring.

```bash
curl "http://127.0.0.1:8000/monitoring/events?limit=100"
```

### `POST /feedback`

Enregistre un retour utilisateur sur la dernière prédiction, sans stocker l'image.

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"verdict":"incorrect","predicted_species":"tomato","predicted_disease":"Late_Blight","corrected_species":"potato","corrected_disease":"Early_Blight"}'
```

Réponse :

```json
{
  "stored": true,
  "message": "Retour enregistré sans conservation de l'image."
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
HF_REPO_ID=DredFury/plant-disease-detection-models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
CONFIDENCE_THRESHOLD=0.65
MONITORING_LOG_PATH=/tmp/plant-disease-detection/predictions.jsonl
```

Le détail du déploiement est décrit dans la page [Déploiement](../deployment.md).
