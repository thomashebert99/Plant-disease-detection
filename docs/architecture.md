# Architecture

Cette page décrit l'architecture réelle du projet : entraînement local, sélection des modèles, publication sur Hugging Face Hub, API FastAPI et interface Streamlit.

## Objectif Fonctionnel

L'utilisateur charge une image de feuille. L'application peut fonctionner en deux modes :

| Mode | Fonctionnement |
|---|---|
| Automatique | L'API détecte l'espèce, vérifie la confiance, puis route vers le modèle maladie de cette espèce |
| Manuel | L'utilisateur indique l'espèce, l'API diagnostique directement la maladie correspondante |

Le mode manuel est important pour l'usage professionnel : un agriculteur connaît souvent l'espèce et veut surtout un diagnostic rapide de la maladie.

## Vue D'ensemble

```text
Utilisateur
  -> Streamlit
      -> API FastAPI
          -> model_loader.py
              -> ensemble_config.json
              -> modèles .keras locaux ou Hugging Face Hub
          -> réponse JSON
      -> affichage résultat
```

En production :

```text
Hugging Face Hub
  stocke les checkpoints sélectionnés et ensemble_config.json

Hugging Face Space API
  héberge FastAPI en Docker
  récupère la config et les checkpoints à la demande depuis Hugging Face Hub

Hugging Face Space Streamlit
  héberge l'interface utilisateur
  appelle l'API publique
```

| Ressource | URL |
|---|---|
| Application Streamlit | `https://dredfury-plant-disease-detection-app.hf.space` |
| API FastAPI | `https://dredfury-plant-disease-detection-api.hf.space` |
| Model repo | `https://huggingface.co/DredFury/plant-disease-detection-models` |

## Pipeline ML

Le pipeline modèle est séparé en trois temps.

### 1. Préparation Des Données

PlantVillage est utilisé pour l'entraînement, la validation et le test in-distribution. PlantDoc est utilisé uniquement comme test out-of-distribution.

```text
data/raw/plantvillage
  -> data/processed/species
  -> data/processed/<species>

data/raw/plantdoc
  -> data/test_ood
```

Les splits sont matérialisés en `train`, `val` et `test`. PlantDoc n'est jamais utilisé pour entraîner les modèles.

### 2. Benchmark Des Modèles

Les notebooks 03 et 04 entraînent plusieurs architectures TensorFlow/Keras :

```text
notebooks/03_benchmark_species.ipynb
  -> tâche species

notebooks/04_benchmark_diseases.ipynb
  -> tâches tomato, apple, grape, corn, potato, pepper, strawberry
```

Chaque tâche suit la même logique :

1. screening rapide avec backbone gelé ;
2. sélection de finalistes ;
3. fine-tuning ;
4. évaluation validation, test in-distribution et PlantDoc OOD quand disponible ;
5. sauvegarde des résultats dans `models/`.

Les checkpoints restent locaux pendant l'entraînement :

```text
models/species/<run>/best_model.keras
models/diseases/<species>/<run>/best_model.keras
```

### 3. Sélection De L'ensemble

Le notebook 05 est la source de vérité de la configuration finale.

```text
notebooks/05_ensemble_selection.ipynb
  -> lit les résultats CSV locaux
  -> sélectionne les meilleurs modèles par tâche
  -> mesure le gain du vote doux
  -> écrit models/ensemble_config.json
```

La sélection ne dépend pas seulement de la validation. Elle tient compte :

- du F1 macro ;
- de la balanced accuracy ;
- de la log loss ;
- du temps d'inférence ;
- du surapprentissage ;
- de l'OOD uniquement lorsque la couverture PlantDoc est fiable ;
- de la diversité raisonnable des familles d'architectures.

La politique finale retenue est `top3_max2_family` : trois modèles sont conservés par tâche, avec au maximum deux modèles d'une même famille sauf si la diversité dégrade trop la qualité. Ce choix garde l'architecture cohérente avec l'objectif initial du projet : l'utilisateur envoie une image, plusieurs modèles votent, puis l'API renvoie une réponse consolidée.

## Modèles Finaux

| Tâche | Classes | Modèles retenus |
|---|---:|---|
| species | 7 | EfficientNetB0, ConvNeXtTiny, EfficientNetB1 |
| tomato | 10 | ConvNeXtTiny, MobileNetV3Large, EfficientNetB0 |
| apple | 4 | EfficientNetB0, MobileNetV3Large, MobileNetV3Small |
| grape | 4 | EfficientNetB1, EfficientNetB0, ConvNeXtTiny |
| corn | 4 | MobileNetV3Large, EfficientNetB1, EfficientNetB0 |
| potato | 3 | MobileNetV3Large, EfficientNetB1, MobileNetV3Small |
| pepper | 2 | MobileNetV3Large, MobileNetV3Small, ConvNeXtTiny |
| strawberry | 2 | MobileNetV3Small, EfficientNetB0, MobileNetV3Large |

Le détail des métriques et des gains du vote doux est disponible dans la page `Résultats`.

## Configuration Des Modèles

`ensemble_config.json` relie les notebooks et l'API. Il évite de coder en dur les modèles sélectionnés dans FastAPI.

Structure simplifiée :

```json
{
  "complete": true,
  "tasks": {
    "species": {
      "task_type": "species",
      "strategy": "soft_vote_mean_probabilities",
      "image_size": 224,
      "class_names": ["apple", "corn", "grape"],
      "models": [
        {
          "run_name": "EfficientNetB0_ft_l50_lr1e_05",
          "architecture": "EfficientNetB0",
          "checkpoint_path": ".../best_model.keras",
          "hub_filename": "models/species/01_EfficientNetB0_....keras"
        }
      ]
    }
  }
}
```

Cette config contient l'ordre exact des classes. C'est essentiel : le modèle renvoie un index, et l'API doit le mapper vers le bon label.

## Vote Doux

Pour une tâche donnée, l'API charge les modèles sélectionnés et moyenne leurs probabilités :

```text
image
  -> modèle 1 -> probabilités
  -> modèle 2 -> probabilités
  -> modèle 3 -> probabilités
  -> moyenne des probabilités
  -> argmax + confiance
```

Le vote doux conserve l'information de confiance, contrairement au vote majoritaire simple.

## API FastAPI

L'API est progressive : elle démarre même si les modèles ne sont pas encore disponibles.

```text
GET  /health
GET  /models/info
GET  /monitoring/summary
POST /predict
POST /predict/species
POST /predict/disease
```

Comportement sans modèles :

| Endpoint | Comportement |
|---|---|
| `/health` | retourne `200` |
| `/models/info` | indique `config_available: false` |
| `/monitoring/summary` | retourne la synthèse des événements déjà loggés |
| `/predict*` | retourne `503` avec un message explicite |

Cela permet de développer, tester et déployer l'API avant la fin des entraînements.

## Model Loader

`src/api/model_loader.py` est la couche centrale entre l'API et TensorFlow.

Il est responsable de :

- lire `ensemble_config.json` ;
- choisir entre modèles locaux et Hugging Face Hub ;
- récupérer les fichiers à la demande depuis Hugging Face Hub si `MODEL_SOURCE=hub` ;
- charger les modèles Keras en lazy loading ;
- gérer les custom objects nécessaires à certaines architectures ;
- faire le vote doux ;
- retourner le label, l'index, la confiance et les probabilités.

TensorFlow est importé tardivement, uniquement au moment de charger un modèle. Ainsi, `/health` et `/models/info` restent rapides et ne chargent pas TensorFlow.

## Preprocessing API

L'API reçoit une image uploadée et la transforme en batch numpy :

```text
bytes image
  -> PIL RGB
  -> resize 224x224
  -> np.float32
  -> shape (1, 224, 224, 3)
```

Il n'y a pas de normalisation côté API. Les modèles construits avec `src/models/build.py` embarquent déjà leur preprocessing Keras.

## Interface Streamlit

Streamlit ne charge aucun modèle. L'interface appelle uniquement l'API.

Responsabilités :

- upload image ;
- choix automatique ou manuel ;
- sélection de l'espèce en mode manuel ;
- appel HTTP vers `/predict` ;
- affichage de l'espèce, de la maladie et des confiances ;
- affichage des trois classes les plus probables pour comprendre les hésitations du modèle ;
- fiche synthétique sur la maladie prédite quand elle est disponible ;
- page séparée pour consulter `/monitoring/summary` ;
- affichage propre des erreurs API.

Cette séparation garde le frontend léger et évite de dupliquer la logique modèle. La page Monitoring est utile pour la démonstration et l'exploitation du service, mais elle reste séparée du parcours utilisateur de diagnostic.

## Déploiement

Le déploiement retenu utilise Hugging Face pour rester simple et gratuit.

| Élément | Plateforme | Rôle |
|---|---|---|
| Modèles `.keras` | Hugging Face Hub | Stockage/versioning des modèles sélectionnés |
| API FastAPI | Hugging Face Space Docker | Service de prédiction public |
| Streamlit | Hugging Face Space Docker | Interface utilisateur publique |

Le Dockerfile API utilise le port `7860`, compatible Hugging Face Spaces, et place le cache HF dans `/tmp/huggingface`.

Le Space Streamlit est aussi lancé en Docker. Il ne contient pas TensorFlow et appelle uniquement l'URL publique de l'API.

## Variables D'environnement

Local :

```env
MODEL_SOURCE=local
ENSEMBLE_CONFIG_PATH=models/ensemble_config.json
CONFIDENCE_THRESHOLD=0.65
MONITORING_LOG_PATH=logs/predictions.jsonl
```

Production API :

```env
MODEL_SOURCE=hub
HF_REPO_ID=DredFury/plant-disease-detection-models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
CONFIDENCE_THRESHOLD=0.65
MONITORING_LOG_PATH=/tmp/plant-disease-detection/predictions.jsonl
```

Production Streamlit :

```env
API_URL=https://dredfury-plant-disease-detection-api.hf.space
```

## Tests Et CI

Les tests sont conçus pour fonctionner sans modèles finaux :

- `/health` ;
- `/models/info` sans config ;
- `/predict` avec config absente ;
- preprocessing image ;
- vote doux avec faux modèles ;
- fonctions d'entraînement et de données.

La CI GitHub Actions exécute :

1. installation CPU ;
2. compilation des entrypoints ;
3. tests ;
4. seuil coverage ;
5. build documentation ;
6. build Docker API sur `main` ou en lancement manuel.

## Monitoring

Le monitoring minimal est implémenté dans `src/monitoring/tracker.py`. À chaque appel de prédiction, l'API écrit un événement JSONL sans stocker l'image. L'endpoint `/monitoring/summary` agrège ensuite :

- le nombre de prédictions ;
- le nombre de réponses `ok`, `uncertain_species` et `error` ;
- la latence moyenne ;
- la confiance moyenne espèce ;
- la confiance moyenne maladie.

Ce choix couvre l'attendu de monitorage du service sans imposer une infrastructure coûteuse ou longue à maintenir pour un projet individuel.

## Flux Final Réalisé

```text
1. Entraîner les modèles dans les notebooks 03 et 04
2. Lancer le notebook 05 pour générer ensemble_config.json
3. Publier les modèles sélectionnés avec scripts/push_models_to_hub.py
4. Déployer/rebuild le Space API
5. Vérifier /health puis /models/info
6. Déployer/rebuild Streamlit
7. Tester un diagnostic complet depuis l'interface
```
