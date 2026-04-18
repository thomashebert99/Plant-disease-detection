# Trame De Rapport

Cette page propose une structure directement réutilisable pour le rapport final. Elle synthétise les choix du projet sans remplacer les détails techniques des autres pages.

## 1. Introduction

Le projet vise à construire une application de diagnostic foliaire à partir d'images. L'objectif est double :

- identifier l'espèce végétale présente sur l'image ;
- diagnostiquer la maladie associée lorsque l'espèce est couverte par les modèles.

L'application finale est composée d'une interface Streamlit, d'une API FastAPI et d'un ensemble de modèles TensorFlow/Keras publiés sur Hugging Face Hub.

Le projet a été réalisé dans un contexte individuel, avec des contraintes fortes de temps, de coût et de ressources matérielles. Ces contraintes ont influencé les choix techniques : modèles sélectionnés plutôt que stockage exhaustif, déploiement gratuit sur Hugging Face Spaces, monitoring minimal et architecture volontairement explicable.

## 2. Données

Deux sources de données sont utilisées.

| Dataset | Usage | Rôle |
|---|---|---|
| PlantVillage | Entraînement, validation, test in-distribution | base principale du projet |
| PlantDoc | Test out-of-distribution | évaluation de robustesse sur images plus proches du terrain |

PlantDoc n'est pas utilisé pour entraîner les modèles. Il sert uniquement à mesurer l'écart entre des images propres et des images plus réalistes.

Point à discuter dans le rapport : les performances PlantVillage peuvent être très élevées car les images sont souvent centrées, propres et prises dans des conditions contrôlées.

## 3. Méthodologie

Le projet a été découpé en cinq étapes ML principales :

1. exploration des données ;
2. préparation des datasets ;
3. benchmark des modèles pour l'espèce ;
4. benchmark des modèles maladie par espèce ;
5. sélection finale d'un ensemble de modèles.

La sélection finale n'utilise pas un seul modèle global. Elle retient trois modèles par tâche, puis applique un vote doux par moyenne des probabilités.

Formulation possible :

> Le choix d'un ensemble permet de stabiliser les prédictions et de conserver une architecture homogène sur toutes les tâches. Même lorsque le gain est faible sur PlantVillage, le vote doux rend la logique de prédiction plus robuste et plus conforme à l'objectif initial du projet.

## 4. Modèles Retenus

La politique finale est `top3_max2_family`.

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

Les métriques détaillées sont dans la page `Résultats`.

## 5. Résultats

Les résultats in-distribution sont très élevés sur PlantVillage. Le vote doux apporte un gain net sur la tâche tomate et reste quasiment neutre sur les tâches déjà saturées.

À retenir :

- meilleure amélioration test : `tomato`, +0.0040 F1 macro ;
- performances proches de 1.0 sur plusieurs tâches PlantVillage ;
- résultats OOD beaucoup plus faibles sur PlantDoc ;
- PlantDoc montre clairement la difficulté du passage vers des images terrain.

Formulation possible :

> Les résultats confirment que le système apprend très bien les distributions PlantVillage, mais que la généralisation à des images terrain reste le principal défi. Le modèle final est donc pertinent pour une démonstration technique, mais il ne doit pas être présenté comme un outil agricole validé en conditions réelles.

## 6. Architecture Applicative

L'application suit une séparation simple :

```text
Streamlit -> FastAPI -> model_loader -> modèles Keras
```

Streamlit ne charge aucun modèle. Il sert uniquement d'interface utilisateur.

FastAPI centralise :

- le preprocessing ;
- le chargement lazy des modèles ;
- la prédiction d'espèce ;
- le diagnostic maladie ;
- le seuil de confiance ;
- le vote doux.

Cette séparation rend le projet plus maintenable et plus facile à déployer.

## 7. Tests Et Monitorage

Les tests couvrent les composants critiques du service :

- endpoints de santé ;
- disponibilité de la configuration modèle ;
- erreurs quand les modèles sont absents ;
- preprocessing image ;
- vote doux ;
- monitoring minimal.

Le monitorage du modèle en service est volontairement simple. À chaque prédiction, l'API écrit un événement JSONL sans stocker l'image uploadée.

L'endpoint suivant permet de démontrer le suivi du service :

```text
GET /monitoring/summary
```

Il expose :

- nombre total de prédictions ;
- nombre de réponses `ok`, incertaines et en erreur ;
- temps de réponse moyen ;
- confiance moyenne espèce ;
- confiance moyenne maladie.

La même synthèse est accessible depuis une page `Monitoring` dans l'interface Streamlit. Elle sert à la démonstration et à l'exploitation du service, sans être mélangée au parcours utilisateur principal.

Formulation possible :

> MLflow est utilisé pour le suivi expérimental des entraînements, tandis que le monitoring de l'API suit les prédictions en service. Cette séparation évite de complexifier le déploiement tout en couvrant les besoins de test et de monitorage du modèle.

## 8. Déploiement

Le déploiement utilise Hugging Face :

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

## 9. Contraintes Projet

Le projet aurait pu être plus simple avec une problématique tabulaire classique, par exemple une prédiction de churn client. Le choix d'un service IA par image ajoute des contraintes importantes :

- entraînement plus coûteux ;
- modèles plus lourds ;
- stockage des checkpoints plus difficile ;
- temps de chargement plus long ;
- déploiement plus sensible aux limites des plateformes gratuites ;
- besoin de séparer clairement modèle, API et interface.

Ces contraintes justifient une architecture pragmatique : Hugging Face Hub pour les modèles finaux, Spaces pour l'API et l'interface, MLflow pour le suivi expérimental, JSONL pour le monitoring minimal.

## 10. Limites

Limites principales à mentionner :

- PlantVillage est un dataset contrôlé ;
- PlantDoc ne couvre pas toutes les classes uniformément ;
- l'OOD reste difficile ;
- les modèles sont nombreux, donc le premier appel API peut être lent ;
- l'application n'a pas été validée par des experts agricoles ;
- le monitoring reste minimal et ne remplace pas une plateforme production complète avec alerting.

## 11. Améliorations Futures

Pistes réalistes :

- ajouter davantage d'images terrain ;
- améliorer les augmentations de données ;
- calibrer les probabilités ;
- ajouter Grad-CAM pour expliquer les prédictions ;
- suivre les erreurs et les prédictions incertaines ;
- optimiser le temps de chargement des modèles ;
- valider le système avec des utilisateurs métier.

## Conclusion Possible

Le projet aboutit à une chaîne complète de diagnostic foliaire : entraînement, sélection d'un ensemble de modèles, API de prédiction, interface utilisateur et déploiement public. Les résultats sont très bons sur PlantVillage, mais les évaluations PlantDoc montrent que la robustesse en conditions réelles reste l'enjeu principal. Le projet constitue donc une base applicative solide et démontrable, avec des limites clairement identifiées et des pistes d'amélioration concrètes.
