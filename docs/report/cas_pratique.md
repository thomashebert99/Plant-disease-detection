# P1 — Cas Pratique

Cette page correspond à la première partie du rapport professionnel : expression du besoin IA, veille technique et réglementaire, benchmark des services existants et paramétrage du service retenu.

## Expression Du Besoin IA

Le projet vise à construire une application web d'aide au diagnostic visuel de maladies foliaires à partir d'une image. L'utilisateur obtient une première indication sur :

- l'espèce végétale détectée ;
- la maladie associée lorsque l'espèce est couverte ;
- un score de confiance ;
- une réponse lisible dans une interface web.

Le système est présenté comme une aide à l'interprétation, pas comme un conseil agronomique certifié.

| Besoin | Contrainte | Implication technique |
|---|---|---|
| Identifier une maladie depuis une image | L'utilisateur ne connaît pas toujours l'espèce | Pipeline en deux étapes : espèce puis maladie |
| Obtenir une réponse rapide | Ressources limitées | Modèles pré-entraînés puis fine-tunés |
| Fournir un résultat compréhensible | Public non expert | Classe prédite, confiance et message de prudence |
| Déployer simplement | Projet individuel | Hugging Face Hub pour les artefacts, Spaces pour l'exposition |
| Comparer plusieurs essais | Plusieurs backbones testés | MLflow/DagsHub pour les runs, métriques et artefacts |
| Limiter la mauvaise généralisation | PlantVillage est un dataset contrôlé | Évaluation OOD sur PlantDoc |
| Limiter les risques sur les données | Images et métadonnées potentiellement sensibles | Secrets hors dépôt, logs minimisés |

## Dispositif De Veille

La veille a été organisée comme un processus de décision : sélectionner des sources fiables, les qualifier, puis transformer l'analyse en recommandation technique.

| Famille de sources | Usage |
|---|---|
| TensorFlow / Keras | Transfer learning et architectures pré-entraînées |
| PlantVillage / PlantDoc | Cadrage dataset, limites terrain et OOD |
| Hugging Face Hub / Spaces | Stockage des artefacts et déploiement |
| MLflow / DagsHub | Suivi expérimental |
| Google / AWS / Azure / Roboflow | Benchmark de services IA managés |
| Kindwise / Pl@ntNet | APIs spécialisées plante |
| Commission européenne / EUR-Lex / CNIL / W3C | RGPD, AI Act, accessibilité |

Les critères de sélection des sources sont l'autorité, l'actualité, la pertinence pour le besoin IA, la vérifiabilité et l'utilité pour une décision de projet.

## Enseignements Techniques Et Réglementaires

Le transfer learning est adapté au projet, car il permet d'exploiter des modèles pré-entraînés sans repartir d'un entraînement from scratch. Plusieurs familles de modèles TensorFlow/Keras ont été étudiées : MobileNetV3, EfficientNet, DenseNet et ConvNeXt.

PlantVillage constitue la base principale d'entraînement, mais ses images sont contrôlées. PlantDoc est donc utilisé comme évaluation out-of-distribution afin de mesurer l'écart avec des images plus proches du terrain.

Sur le plan réglementaire et éthique :

- le RGPD devient applicable si l'image ou ses métadonnées permettent d'identifier une personne ;
- les logs sont limités aux informations utiles au suivi du service ;
- les secrets sont stockés hors dépôt ;
- le score de confiance est présenté comme un indicateur, pas comme une probabilité calibrée ;
- l'AI Act est abordé avec prudence : le projet ne semble pas relever a priori d'une catégorie high-risk standard, mais cette qualification dépend du contexte exact d'usage.

## Inventaire Des Services IA Existants

| Type | Services identifiés | Caractéristiques |
|---|---|---|
| Bibliothèques open-source | TensorFlow/Keras, Keras Applications | Contrôle fort, fine-tuning, besoin d'ingénierie |
| Services AutoML cloud | Vertex AI, Azure Custom Vision, AWS Rekognition Custom Labels | Entraînement et serving managés, coût récurrent |
| Plateformes vision | Roboflow | Annotation, entraînement, hosted ou self-hosted |
| APIs spécialisées plante | plant.health / crop.health, Pl@ntNet diseases | Intégration rapide, taxonomie externe, faible contrôle |
| Plateformes MLOps / déploiement | Hugging Face Hub/Spaces, MLflow/DagsHub | Artefacts, démonstration, tracking |

## Benchmark Et Recommandation

Le benchmark ne compare pas uniquement des produits équivalents, mais les options réalistes disponibles pour répondre au besoin IA dans les contraintes du projet.

| Solution | Testée dans le projet ? | Contrôle modèle | Coût | Déploiement | Décision |
|---|---:|---|---|---|---|
| TensorFlow/Keras + Hugging Face + MLflow/DagsHub | Oui | Fort | Faible à modéré | Moyen | Retenue |
| Azure AI Custom Vision | Benchmark documentaire | Moyen | Moyen | Simple | Alternative managée |
| Vertex AI AutoML Vision | Benchmark documentaire | Moyen | Plus élevé | Robuste | Alternative entreprise |
| AWS Rekognition Custom Labels | Benchmark documentaire | Moyen | Plus élevé à l'usage | Simple à moyen | Alternative crédible |
| Roboflow | Benchmark documentaire | Moyen | Variable | Simple | Alternative vision ops |
| plant.health / crop.health | Benchmark documentaire / API externe | Faible | À la requête | Très simple | Comparaison métier |
| Pl@ntNet diseases | Benchmark documentaire / API externe | Faible | Quota / crédits | Simple | Comparaison spécialisée |

Un benchmark expérimental complet multi-plateforme n'a pas été retenu, car il aurait demandé de reconstruire les datasets, taxonomies, entraînements et endpoints sur plusieurs plateformes, avec un coût et un délai incompatibles avec le projet.

La solution retenue combine :

- TensorFlow/Keras Applications pour les modèles pré-entraînés et le fine-tuning ;
- Hugging Face Hub pour l'hébergement et le versioning des artefacts ;
- Hugging Face Spaces pour l'exposition de l'API et de l'interface ;
- MLflow/DagsHub pour le suivi expérimental ;
- JSONL pour un monitoring léger des prédictions.

## Paramétrage Du Service Retenu

Les services effectivement paramétrés sont Hugging Face Hub, Hugging Face Spaces et MLflow/DagsHub.

| Variable | Rôle |
|---|---|
| `MLFLOW_TRACKING_URI` | URI du tracking MLflow/DagsHub |
| `MLFLOW_TRACKING_USERNAME` | Identifiant de connexion |
| `MLFLOW_TRACKING_PASSWORD` | Token ou mot de passe d'accès |
| `CONFIDENCE_THRESHOLD` | Seuil de confiance pour l'espèce |
| `MODEL_SOURCE` | Source de chargement : `local` ou `hub` |
| `ENSEMBLE_CONFIG_PATH` | Chemin de configuration locale |
| `MONITORING_LOG_PATH` | Chemin du fichier JSONL |
| `HF_TOKEN` | Token Hugging Face |
| `HF_REPO_ID` | Dépôt Hugging Face des artefacts modèles |
| `API_URL` | URL de l'API appelée par Streamlit |

Commandes principales :

```bash
make run-api
make run-app
```

Vérifications fonctionnelles :

| Vérification | Preuve attendue |
|---|---|
| API disponible | `/health` |
| Documentation API | `/docs` |
| Configuration modèles | `/models/info` |
| Prédiction image | Réponse JSON |
| Interface disponible | Upload et résultat Streamlit |
| Tracking actif | Run MLflow/DagsHub |
| Monitoring actif | Extrait JSONL anonymisé |

## Sources

- PlantVillage : [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/plant_village)
- PlantDoc : [arXiv 1911.10317](https://arxiv.org/abs/1911.10317)
- Biais PlantVillage : [arXiv 2206.04374](https://arxiv.org/abs/2206.04374)
- Keras Applications : [Keras](https://keras.io/api/applications/)
- Hugging Face Spaces : [Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview)
- MLflow Tracking : [MLflow](https://mlflow.org/docs/latest/ml/tracking/)
- AI Act : [EUR-Lex 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)
