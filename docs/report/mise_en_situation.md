# P2 — Mise En Situation

Cette page correspond à la deuxième partie du rapport professionnel : réalisation du pipeline, entraînement, sélection, intégration applicative, tests, déploiement et limites.

## Objectif Du Système

L'application finale répond à deux questions :

1. quelle espèce végétale est visible sur l'image ;
2. quelle maladie est détectée pour cette espèce, lorsque la tâche est couverte.

Le système complet associe :

- des notebooks d'exploration, d'entraînement et de sélection ;
- une configuration finale d'ensemble ;
- une API FastAPI ;
- une interface Streamlit ;
- un stockage des artefacts sur Hugging Face Hub ;
- un déploiement sur Hugging Face Spaces ;
- un monitoring léger par logs JSONL.

## Données

| Dataset | Usage | Rôle |
|---|---|---|
| PlantVillage | Entraînement, validation, test in-distribution | Base principale du projet |
| PlantDoc | Test out-of-distribution | Diagnostic de robustesse sur images plus proches du terrain |

PlantDoc n'est pas utilisé pour entraîner les modèles. Il sert uniquement à mesurer l'écart entre des images contrôlées et des images plus réalistes.

## Méthodologie

Le projet a été découpé en cinq étapes principales :

1. exploration des données ;
2. préparation des datasets ;
3. benchmark des modèles pour l'espèce ;
4. benchmark des modèles maladie par espèce ;
5. sélection finale d'un ensemble de modèles.

La sélection finale retient trois modèles par tâche avec la stratégie `top3_max2_family`, puis applique un vote doux par moyenne des probabilités.

## Modèles Retenus

| Tâche | Modèles retenus |
|---|---|
| species | EfficientNetB0, ConvNeXtTiny, EfficientNetB1 |
| tomato | ConvNeXtTiny, MobileNetV3Large, EfficientNetB0 |
| apple | EfficientNetB0, MobileNetV3Large, MobileNetV3Small |
| grape | EfficientNetB1, EfficientNetB0, ConvNeXtTiny |
| corn | MobileNetV3Large, EfficientNetB1, EfficientNetB0 |
| potato | MobileNetV3Large, EfficientNetB1, MobileNetV3Small |
| pepper | MobileNetV3Large, MobileNetV3Small, ConvNeXtTiny |
| strawberry | MobileNetV3Small, EfficientNetB0, MobileNetV3Large |

## Résultats

Les résultats in-distribution sont très élevés sur PlantVillage. Le vote doux apporte un gain net sur la tâche tomate et reste quasiment neutre sur les tâches déjà saturées.

À retenir :

- meilleure amélioration test : `tomato`, +0.0040 F1 macro ;
- performances proches de 1.0 sur plusieurs tâches PlantVillage ;
- résultats OOD beaucoup plus faibles sur PlantDoc ;
- PlantDoc confirme la difficulté du passage vers des images terrain.

## Architecture Applicative

```text
Streamlit -> FastAPI -> model_loader -> modèles Keras
```

Streamlit ne charge aucun modèle. Il sert d'interface utilisateur et appelle l'API.

FastAPI centralise :

- le preprocessing ;
- le chargement lazy des modèles ;
- la prédiction d'espèce ;
- le diagnostic maladie ;
- le seuil de confiance ;
- le vote doux.

## Tests Et Monitoring

Les tests couvrent les composants critiques :

- endpoints de santé ;
- disponibilité de la configuration modèle ;
- erreurs quand les modèles sont absents ;
- preprocessing image ;
- vote doux ;
- monitoring minimal.

Le monitoring de service est volontairement simple. À chaque prédiction, l'API écrit un événement JSONL sans stocker l'image uploadée.

L'endpoint suivant expose la synthèse :

```text
GET /monitoring/summary
```

Il retourne notamment :

- nombre total de prédictions ;
- nombre de réponses `ok`, incertaines et en erreur ;
- latence moyenne ;
- confiance moyenne espèce ;
- confiance moyenne maladie.

MLflow est utilisé pour le suivi expérimental des entraînements, tandis que le monitoring JSONL suit les prédictions en service.

## Déploiement

| Élément | Plateforme |
|---|---|
| Checkpoints `.keras` | Hugging Face Hub |
| API FastAPI | Hugging Face Space Docker |
| Interface Streamlit | Hugging Face Space Docker |

URLs :

```text
https://dredfury-plant-disease-detection-app.hf.space
https://dredfury-plant-disease-detection-api.hf.space
https://huggingface.co/DredFury/plant-disease-detection-models
```

## Limites

- PlantVillage est un dataset contrôlé ;
- PlantDoc ne couvre pas toutes les classes uniformément ;
- l'OOD reste difficile ;
- les modèles sont nombreux, donc le premier appel API peut être lent ;
- l'application n'a pas été validée par des experts agricoles ;
- le monitoring reste minimal et ne remplace pas une plateforme complète d'observabilité production.

## Conclusion

Le projet aboutit à une chaîne complète de diagnostic foliaire : entraînement, sélection d'un ensemble de modèles, API de prédiction, interface utilisateur et déploiement public. Les résultats sont très bons sur PlantVillage, mais les évaluations PlantDoc montrent que la robustesse en conditions réelles reste l'enjeu principal.
