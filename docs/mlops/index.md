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

Après la sélection finale, le script `scripts/log_final_selection_to_mlflow.py` permet de créer un run MLflow récapitulatif sans relancer d'entraînement. Il lit `ensemble_config.json` et les CSV du dossier `models/ensemble/`, puis logge les paramètres de sélection, les métriques globales et les artefacts de décision.

```bash
make log-final-selection-dry-run
make log-final-selection
```

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

## Monitoring du service IA

Le monitoring reste volontairement léger, mais couvre maintenant trois niveaux : santé de l'API, fiabilité du modèle et signaux de drift en vision par ordinateur.

À chaque appel de prédiction, l'API écrit un événement JSONL via `src/monitoring/tracker.py`. L'image uploadée n'est jamais stockée. Seules des informations dérivées sont conservées :

- timestamp, endpoint et mode automatique ou manuel ;
- statut `ok`, `uncertain_species` ou `error` ;
- espèce et maladie prédites quand disponibles ;
- confiances espèce et maladie ;
- latence et source des modèles ;
- métriques image non ré-identifiantes : luminosité, contraste, netteté approximative, saturation, ratio vert/brun, taille et ratio d'aspect.

L'endpoint principal expose une synthèse démontrable :

```text
GET /monitoring/summary
```

Il retourne notamment :

- nombre total de prédictions ;
- taux `ok`, `uncertain_species`, `error` et faible confiance ;
- latence moyenne, minimum, maximum et P95 ;
- confiance moyenne espèce et maladie ;
- distributions des espèces et maladies prédites ;
- histogrammes de confiance ;
- alertes actives ;
- synthèse des retours utilisateur ;
- état de drift du flux récent ;
- signal de dérive qualité issu du taux de désaccord utilisateur.

Un second endpoint expose les derniers événements pour alimenter les graphes temporels :

```text
GET /monitoring/events?limit=100
```

### Détection du drift sans stockage des images

Le drift est détecté par comparaison entre une fenêtre récente de production et deux références :

- `plantvillage_in_domain` : domaine d'entraînement/validation contrôlé ;
- `plantdoc_ood` : domaine OOD connu, plus proche des conditions terrain.

Cette double référence évite de traiter automatiquement toute image OOD comme une erreur. Le dashboard distingue :

- `in_domain` : flux proche de PlantVillage ;
- `ood_like` : flux proche du domaine OOD connu PlantDoc, donc à surveiller ;
- `reference_shift` : flux éloigné des références mais encore explicable ;
- `unknown_shift` : décalage fort et non couvert par les références connues.

Les signaux utilisés sont des proxys : métriques image, confiances et distributions de prédictions. Ils ne remplacent pas une mesure de performance avec vérité terrain, mais ils permettent de déclencher une surveillance, une campagne d'annotation ou une analyse complémentaire.

### Feedback utilisateur

L'interface Streamlit propose un retour après prédiction : correcte, incorrecte ou incertaine. L'endpoint `POST /feedback` stocke ce retour dans un JSONL séparé, sans image. Ce feedback permet de suivre un taux de désaccord et d'identifier les classes à prioriser pour une amélioration future.

Le feedback ne détecte pas le data drift au sens strict : il détecte plutôt une possible dérive de qualité ou de concept, car il apporte une vérité terrain utilisateur. Le dashboard l'affiche donc comme un signal complémentaire `model_quality_shift`. S'il y a à la fois un flux OOD/inconnu et beaucoup de désaccords, le risque devient beaucoup plus crédible.

### Réentraînement hypothétique

Le réentraînement sur images utilisateur n'est pas implémenté. Dans une version production, il nécessiterait un consentement explicite séparé, une durée de conservation définie, la suppression des métadonnées EXIF, un accès restreint, une file d'annotation et un droit de suppression. Le projet se limite donc à la détection et à la restitution des signaux utiles à l'amélioration itérative.

### Workflow de monitoring

```text
Image utilisateur
  -> prédiction API
  -> métriques dérivées sans stockage image
  -> JSONL prédictions
  -> agrégation /monitoring/summary
  -> dashboard Streamlit
  -> alertes, drift, feedback
  -> priorisation d'une amélioration future
```

Ce choix est adapté à la certification : il rend le modèle observable sans imposer une infrastructure coûteuse ou longue à maintenir pour un projet individuel.

## Contraintes Du Projet

Le projet a été réalisé seul, avec des contraintes de temps, de coût et de ressources matérielles.

Ces contraintes expliquent plusieurs choix :

- entraînement local plutôt que cloud GPU massif ;
- sélection contrôlée de 24 checkpoints finaux plutôt qu'un stockage exhaustif de tous les modèles ;
- Hugging Face Hub pour stocker uniquement les artefacts nécessaires à l'API ;
- Spaces gratuits pour exposer l'API et l'interface ;
- monitoring JSONL enrichi plutôt qu'une stack Prometheus/Grafana ou un outil payant ;
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
