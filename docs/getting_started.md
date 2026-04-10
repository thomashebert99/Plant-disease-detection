# Getting Started

## Prérequis

- Python `3.11.9`
- `venv`
- `pip`

## Environnement local

Le projet fixe Python via [`.python-version`](/home/thomashebert99/code/thomashebert99/Plant-disease-detection/.python-version) et utilise un virtualenv local `.venv`.

```bash
pyenv install 3.11.9
pyenv local 3.11.9
make install
```

## Variantes d'installation

```bash
make install      # base locale CPU + tests + notebooks + docs
make install-gpu  # variante locale GPU TensorFlow (RTX 4090)
make test
```

## Jeux de donnees

- `data/processed/` contient toutes les classes retenues du projet a partir de PlantVillage.
- `data/test_ood/` contient uniquement le sous-ensemble de classes reellement disponibles dans PlantDoc apres alignement.
- Une classe absente de PlantDoc n'est pas retiree du projet ni de l'entrainement : cela signifie seulement qu'elle n'est pas encore evaluee en OOD avec ce dataset.
- Si un autre dataset ou des images perso couvrent ces classes plus tard, elles pourront etre ajoutees a l'evaluation sans changer le perimetre du modele.

## Sans Makefile

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-cpu.txt -r requirements-dev.txt
```

## Conteneur API

```bash
docker compose up --build
```

L'API FastAPI sera disponible sur `http://localhost:8000`.
