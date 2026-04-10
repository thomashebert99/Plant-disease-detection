VENV ?= .venv-plant-disease-detection
PYTHON ?= python3.11
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: venv install install-gpu test docs run-api run-app verify-gpu push-models clean-venv

venv:             ## Créer le virtualenv local si besoin
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel

install: venv     ## Installation locale CPU simple
	$(PIP) install --default-timeout=120 --retries 10 -r requirements-cpu.txt -r requirements-dev.txt

install-gpu: venv ## Installation locale GPU TensorFlow (WSL2 + RTX 4090)
	$(PIP) install --default-timeout=120 --retries 10 -r requirements-gpu.txt -r requirements-dev.txt

test:             ## Lancer les tests
	$(VENV_PYTHON) -m pytest

docs:             ## Lancer la doc en local
	$(VENV)/bin/mkdocs serve

run-api:          ## Lancer l'API FastAPI
	$(VENV)/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

run-app:          ## Lancer l'interface Streamlit
	$(VENV)/bin/streamlit run app/streamlit_app.py

verify-gpu:       ## Vérifier la détection GPU TensorFlow
	$(VENV_PYTHON) -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

push-models:      ## Uploader les modèles entraînés sur HF Hub
	$(VENV_PYTHON) scripts/push_models_to_hub.py

clean-venv:       ## Supprimer le virtualenv local
	rm -rf $(VENV)
