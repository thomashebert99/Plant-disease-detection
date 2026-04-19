# Plant Disease Detection

Bienvenue dans la documentation technique du projet de diagnostic foliaire par image.

Le projet combine :

- des notebooks pour préparer les données, entraîner les modèles et sélectionner l'ensemble final ;
- une API FastAPI pour exposer les prédictions ;
- une interface Streamlit pour tester le diagnostic avec une image ;
- un déploiement sur Hugging Face Hub et Hugging Face Spaces.

## Démo en ligne

| Service | URL |
|---|---|
| Application Streamlit | `https://dredfury-plant-disease-detection-app.hf.space` |
| API FastAPI | `https://dredfury-plant-disease-detection-api.hf.space` |
| Model repo Hugging Face | `https://huggingface.co/DredFury/plant-disease-detection-models` |

Endpoints de vérification :

```text
https://dredfury-plant-disease-detection-api.hf.space/health
https://dredfury-plant-disease-detection-api.hf.space/models/info
https://dredfury-plant-disease-detection-api.hf.space/monitoring/summary
```

## Pages principales

| Page | Contenu |
|---|---|
| Guide de démarrage | Installation locale, commandes utiles et lancement Docker |
| Architecture | Organisation du pipeline ML, de l'API et de l'interface |
| Résultats | Modèles retenus, gains du vote doux et limites OOD |
| Déploiement | Docker Compose, Hugging Face Hub et Hugging Face Spaces |
| API | Endpoints, schémas de réponse et exemples d'appel |
| MLOps | Suivi d'expériences, tests et automatisation |
| Rapport de projet - P1 | Cas pratique : veille, benchmark et paramétrage |
| Rapport de projet - P2 | Mise en situation : réalisation, résultats et déploiement |
