# MLOps

Cette page résume les pratiques MLOps réellement utiles pour ce projet : reproductibilité des expériences, gestion des artefacts, tests, CI/CD et déploiement.

## Objectif

Le projet n'a pas besoin d'une plateforme MLOps complexe. L'objectif est plutôt d'avoir une chaîne simple et défendable :

```text
notebooks -> artefacts modèles -> configuration finale -> API -> interface -> documentation
```

Chaque étape doit produire un fichier clair ou un service vérifiable.

## Artefacts Modèle

Les notebooks écrivent les résultats dans `models/`.

| Artefact | Produit par | Utilisation |
|---|---|---|
| Checkpoints `.keras` | notebooks 03 et 04 | poids des modèles entraînés |
| CSV de benchmarks | notebooks 03 et 04 | comparaison des architectures |
| `models/ensemble_config.json` | notebook 05 | configuration finale lue par l'API locale |
| `models/ensemble/ensemble_config_hf.json` | notebook 05 + script upload | configuration avec chemins Hugging Face Hub |
| Repo HF `DredFury/plant-disease-detection-models` | `scripts/push_models_to_hub.py` | stockage public ou privé des modèles finaux |

Le repo applicatif ne doit pas contenir les gros checkpoints. Les poids sont publiés sur Hugging Face Hub ; en mode `hub`, l'API récupère la configuration et les checkpoints à la demande, puis met les modèles chargés en cache mémoire.

## Suivi Des Expériences

Le suivi combine MLflow/DagsHub et des fichiers simples :

- CSV de métriques par modèle ;
- dossiers de runs dans `models/` ;
- runs MLflow/DagsHub pour tracer paramètres, métriques et artefacts utiles à la comparaison des essais ;
- notebook 05 pour la décision finale ;
- page `Résultats` pour la synthèse.

Ce choix est adapté au calendrier du projet : il reste compréhensible, versionnable et facile à expliquer dans le rapport. MLflow/DagsHub couvre le suivi expérimental des entraînements et benchmarks ; le monitoring du service déployé reste séparé et repose sur les logs JSONL exposés par `/monitoring/summary`.

## Sélection Finale

La stratégie retenue est `top3_max2_family`.

Elle répond à deux contraintes :

- garder trois modèles par tâche pour respecter l'idée d'ensemble et de vote ;
- éviter de choisir trois variantes trop proches si une famille domine légèrement le classement.

Le vote doux est ensuite appliqué partout :

```text
probabilité finale = moyenne des probabilités des 3 modèles
```

Cette logique est simple à expliquer et stable côté API.

## Tests

Les tests sont conçus pour ne pas dépendre des checkpoints finaux.

Ils couvrent notamment :

- démarrage API ;
- `/health` ;
- `/models/info` avec et sans configuration ;
- preprocessing image ;
- vote doux avec modèles factices ;
- comportement clair quand les modèles sont absents ;
- résolution des chemins locaux dans Docker.

Commande :

```bash
make test
```

## CI/CD

Le workflow GitHub Actions vérifie le projet automatiquement :

1. installation des dépendances CPU ;
2. compilation des principaux entrypoints Python ;
3. tests ;
4. seuil minimal de coverage ;
5. build strict de la documentation ;
6. build Docker API sur `main` ou en lancement manuel.

La CI ne télécharge pas les gros modèles. C'est volontaire : elle valide le code, pas l'inférence complète avec les checkpoints.

## Déploiement

Le déploiement est séparé en trois ressources :

| Ressource | Plateforme | Rôle |
|---|---|---|
| Modèles | Hugging Face Hub | stockage des artefacts ML |
| API | Hugging Face Space Docker | inférence et endpoints HTTP |
| Interface | Hugging Face Space Docker | upload image et affichage utilisateur |

Le découpage évite de mettre TensorFlow dans Streamlit et garde une séparation claire entre frontend et backend.

## Monitoring Minimal

Le monitoring de service est implémenté de façon volontairement légère.

À chaque appel de prédiction, l'API écrit un événement JSONL via `src/monitoring/tracker.py`. L'image uploadée n'est jamais stockée.

Champs suivis :

- timestamp ;
- endpoint ;
- mode automatique ou manuel ;
- statut `ok`, `uncertain_species` ou `error` ;
- espèce et maladie prédites quand disponibles ;
- confiances ;
- temps de réponse ;
- source des modèles, locale ou Hugging Face Hub.

L'endpoint suivant expose une synthèse démontrable :

```text
GET /monitoring/summary
```

Il retourne notamment :

- nombre total de prédictions ;
- nombre d'erreurs ;
- nombre de prédictions incertaines ;
- latence moyenne ;
- confiance moyenne espèce ;
- confiance moyenne maladie.

L'interface Streamlit expose aussi une page `Monitoring` qui appelle cet endpoint. Cela facilite la démonstration devant un jury, tout en gardant la supervision séparée de l'écran de diagnostic.

Ce choix est adapté à la certification : il montre que le modèle en service est observable sans imposer une infrastructure de monitoring coûteuse.

## Contraintes Du Projet

Le projet a été réalisé seul, avec des contraintes de temps, de coût et de ressources matérielles.

Ces contraintes expliquent plusieurs choix :

- entraînement local plutôt que cloud GPU massif ;
- sélection contrôlée de 24 checkpoints finaux plutôt qu'un stockage exhaustif de tous les modèles ;
- Hugging Face Hub pour stocker uniquement les artefacts nécessaires à l'API ;
- Spaces gratuits pour exposer l'API et l'interface ;
- monitoring JSONL minimal plutôt qu'une stack Prometheus/Grafana ou un outil payant ;
- MLflow utilisé pour le suivi expérimental, pas comme plateforme de monitoring du service déployé.

Un projet plus simple de machine learning tabulaire aurait demandé moins de calcul et moins de stockage. Ici, la difficulté vient du passage à un service IA complet : modèles lourds, API, application, packaging, déploiement, tests et monitorage.

## Sécurité

Points importants :

- `.env` reste local et ignoré par Git ;
- les tokens Hugging Face sont des secrets, jamais des variables publiques ;
- un token affiché par erreur doit être révoqué puis remplacé ;
- le Space Streamlit ne doit pas connaître le token modèle ;
- seul le Space API a besoin de `HF_TOKEN` si le repo modèle est privé.

## Limites MLOps

Les principales limites du dispositif actuel :

- pas de réentraînement automatique ;
- pas de monitoring production complet avec alerting ;
- pas de validation métier réelle ;
- dépendance au démarrage à froid de Hugging Face Spaces ;
- modèles nombreux, donc premier chargement potentiellement lent.

Ces limites sont acceptables pour un projet de démonstration complet, à condition d'être clairement discutées.
