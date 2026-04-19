VENV ?= .venv-plant-disease-detection
PYTHON ?= python3.11
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUN_PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))

.PHONY: venv install install-gpu test docs run-api run-app verify-gpu push-models log-final-selection-dry-run log-final-selection clean-venv

venv:             ## Créer le virtualenv local si besoin
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel

install: venv     ## Installation locale CPU simple
	$(PIP) install --default-timeout=120 --retries 10 -r requirements-cpu.txt -r requirements-dev.txt

install-gpu: venv ## Installation locale GPU TensorFlow (WSL2 + RTX 4090)
	$(PIP) install --default-timeout=120 --retries 10 -r requirements-gpu.txt -r requirements-dev.txt

test:             ## Lancer les tests
	$(RUN_PYTHON) -m pytest

docs:             ## Lancer la doc en local
	$(RUN_PYTHON) -m mkdocs serve

run-api:          ## Lancer l'API FastAPI
	$(RUN_PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

run-app:          ## Lancer l'interface Streamlit
	$(RUN_PYTHON) -m streamlit run app/streamlit_app.py

verify-gpu:       ## Vérifier la détection GPU TensorFlow
	$(RUN_PYTHON) -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

push-models:      ## Uploader les modèles entraînés sur HF Hub
	$(RUN_PYTHON) scripts/push_models_to_hub.py

log-final-selection-dry-run: ## Prévisualiser le run MLflow de sélection finale
	$(RUN_PYTHON) scripts/log_final_selection_to_mlflow.py --dry-run

log-final-selection: ## Logger la sélection finale dans MLflow/DagsHub
	$(RUN_PYTHON) scripts/log_final_selection_to_mlflow.py

clean-venv:       ## Supprimer le virtualenv local
	rm -rf $(VENV)
