# Déploiement Et Docker

Cette page décrit le lancement local avec Docker Compose et le déploiement public sur Hugging Face Hub et Hugging Face Spaces.

## Ressources En Ligne

| Ressource | Rôle | URL publique | Page Hugging Face |
|---|---|---|---|
| Model repo | Stockage de `ensemble_config.json` et des checkpoints `.keras` | `https://huggingface.co/DredFury/plant-disease-detection-models` | `https://huggingface.co/DredFury/plant-disease-detection-models` |
| Space API | Hébergement FastAPI en Docker | `https://dredfury-plant-disease-detection-api.hf.space` | `https://huggingface.co/spaces/DredFury/Plant-disease-detection-api` |
| Space Streamlit | Interface utilisateur | `https://dredfury-plant-disease-detection-app.hf.space` | `https://huggingface.co/spaces/DredFury/Plant-disease-detection-app` |

Le choix Hugging Face est volontairement simple : un repo pour les artefacts ML, un Space pour l'API, un Space pour l'interface.

## Variables Du Space API

Dans le Space API, ouvrir `Settings` puis `Variables and secrets`.

Variables :

```env
MODEL_SOURCE=hub
HF_REPO_ID=DredFury/plant-disease-detection-models
CONFIDENCE_THRESHOLD=0.65
MONITORING_STORAGE_DIR=/data/plant-disease-detection/monitoring
```

Secret :

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

`HF_TOKEN` doit avoir un droit de lecture sur le repo modèle si le repo est privé. Il ne doit jamais être commité dans Git.

## Stockage Persistant Du Monitoring

Dans le Space API, ouvrir `Settings`, puis `Storage Buckets`, et monter un bucket en lecture-écriture :

| Paramètre | Valeur |
|---|---|
| Bucket | `DredFury/plant-disease-monitoring` |
| Mount path | `/data` |
| Access mode | `Read & Write` |

Le bucket peut être privé si l'option est disponible. Le mode public reste acceptable pour ce prototype parce que les images utilisateur ne sont pas stockées, mais un bucket privé est préférable dès que le service sort d'une démonstration.

Le montage redémarre le Space API. Après redémarrage, l'API écrit automatiquement :

```text
/data/plant-disease-detection/monitoring/predictions.jsonl
/data/plant-disease-detection/monitoring/feedback.jsonl
```

Ces fichiers contiennent les événements de prédiction et les retours utilisateur sans image brute.

## Variables Du Space Streamlit

Dans le Space Streamlit, ouvrir `Settings` puis `Variables and secrets`.

Variable :

```env
API_URL=https://dredfury-plant-disease-detection-api.hf.space
```

Streamlit ne charge aucun modèle. Il envoie les images à l'API et affiche la réponse.

## Docker Local

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
CONFIDENCE_THRESHOLD=0.65
MONITORING_STORAGE_DIR=/app/logs/monitoring
```

Le dossier local `models/` est monté dans le conteneur API en lecture seule. Cela évite de copier les checkpoints dans l'image Docker. Le dossier `logs/` est monté en écriture pour conserver le JSONL de monitoring entre deux redémarrages locaux.

Streamlit appelle l'API avec l'URL interne du réseau Docker :

```env
API_URL=http://api:7860
```

## Docker API Seul

Le `Dockerfile` API est compatible Hugging Face Spaces :

- l'API écoute sur le port `7860` par défaut ;
- `HF_HOME=/data/.huggingface` place le cache Hugging Face dans le volume monté sur `/data` quand un stockage persistant est attaché au Space ;
- `MONITORING_STORAGE_DIR=/data/plant-disease-detection/monitoring` écrit les JSONL de prédiction et de feedback dans le même volume ;
- `MODEL_SOURCE=hub` indique à l'API de récupérer la configuration et les checkpoints depuis le Hub à la demande, puis de mettre les modèles chargés en cache mémoire.

Sur Hugging Face Spaces, le système de fichiers par défaut reste éphémère. Pour conserver le monitoring après sommeil ou redémarrage, il faut attacher le Storage Bucket `DredFury/plant-disease-monitoring` en lecture-écriture sur `/data` dans les settings du Space API.

En local, on peut surcharger le port :

```bash
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 -e PORT=8000 plant-disease-api
```

## Docker Streamlit Seul

Le `Dockerfile.streamlit` lance l'interface Streamlit sur le port défini dans le conteneur. En local, le Docker Compose expose l'interface sur `http://localhost:8501`.

Sur Hugging Face Spaces, l'upload de fichiers depuis le navigateur peut déclencher des erreurs liées aux protections CORS/XSRF. Le démarrage Streamlit désactive donc explicitement ces protections dans le conteneur :

```bash
streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableXsrfProtection false \
  --server.enableCORS false
```

Sur Hugging Face Spaces, le port exposé doit rester cohérent entre le Dockerfile, la configuration du Space et la commande de démarrage. Ce réglage concerne uniquement l'interface Streamlit. L'API FastAPI reste séparée.

## Publication Des Modèles

Après la fin des notebooks 04 puis 05, le fichier suivant doit exister :

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

Si une coupure réseau interrompt l'upload, relancer à partir du dernier fichier en erreur :

```bash
python scripts/push_models_to_hub.py \
  --start-at models/pepper/03_ConvNeXtTiny_ConvNeXtTiny_ft_l50_lr1e_05.keras \
  --max-retries 5 \
  --retry-wait-seconds 30
```

Le nom passé à `--start-at` doit correspondre au chemin Hugging Face indiqué dans l'erreur.

## Déploiement Des Spaces

Chaque Space est un repo Git Hugging Face. Le principe est :

1. cloner le repo du Space API ;
2. copier le code nécessaire à l'API : `Dockerfile`, `requirements-api.txt`, `requirements-api-cpu.txt`, `src/`, `README.md` si souhaité ;
3. commiter et pousser vers le Space API ;
4. attendre la fin du build Hugging Face ;
5. vérifier `/health` puis `/models/info` ;
6. cloner le repo du Space Streamlit ;
7. copier le code nécessaire à l'interface : `Dockerfile.streamlit`, `requirements-streamlit.txt`, `app/`, `README.md` si souhaité ;
8. commiter et pousser vers le Space Streamlit ;
9. tester l'upload d'une image dans l'interface.

Le build se déclenche automatiquement à chaque `git push`.

Si Git refuse le mot de passe, c'est normal : Hugging Face demande un token ou une clé SSH. La méthode simple est :

```bash
hf auth login
git config --global credential.helper store
```

Ensuite, utiliser le nom de compte Hugging Face comme username et le token comme password quand Git les demande.

## Vérifications API

Vérifier que l'API répond :

```text
https://dredfury-plant-disease-detection-api.hf.space/health
```

Résultat attendu :

```json
{"status": "ok"}
```

Vérifier que les modèles sont disponibles :

```text
https://dredfury-plant-disease-detection-api.hf.space/models/info
```

Résultat attendu :

- `config_available: true` ;
- `complete: true` ;
- `model_count: 3` pour chaque tâche ;
- aucune tâche manquante.

Si `config_available` vaut `false`, vérifier :

- que `MODEL_SOURCE=hub` est bien défini dans le Space API ;
- que `HF_REPO_ID` vaut `DredFury/plant-disease-detection-models` ;
- que `HF_TOKEN` a accès au repo si celui-ci est privé ;
- que `scripts/push_models_to_hub.py` a bien uploadé `ensemble_config.json`.

Vérifier le monitoring :

```text
https://dredfury-plant-disease-detection-api.hf.space/monitoring/summary
```

Après quelques prédictions, `total_predictions` doit augmenter. Avec un stockage persistant monté sur `/data`, les fichiers `/data/plant-disease-detection/monitoring/predictions.jsonl` et `/data/plant-disease-detection/monitoring/feedback.jsonl` sont conservés après sommeil ou redémarrage du Space API.

## Points De Vigilance

- Le premier diagnostic peut être lent : les modèles sont chargés en lazy loading.
- Hugging Face Spaces gratuits peuvent dormir après inactivité, donc le premier appel peut aussi réveiller le Space.
- Les checkpoints sont volumineux : le repo modèle doit rester la source des poids, pas le repo applicatif.
- Le monitoring JSONL devient persistant seulement si `/data` est réellement un volume/bucket persistant du Space API ; sans volume attaché, il reste local au conteneur.
- `.env` doit rester local et ignoré par Git.
- Si un token Hugging Face a été affiché par erreur, il faut le révoquer et en créer un nouveau.
