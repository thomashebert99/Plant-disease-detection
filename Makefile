.PHONY: install install-gpu install-cpu test lint format docs

install:          ## Installation standard CPU (cloud-compatible)
	poetry install --with dev,docs

install-gpu:      ## Installation GPU locale (WSL2 + RTX 4090)
	poetry install --with dev,docs
	pip install "tensorflow[and-cuda]==2.19.0"
	pip install torch==2.3.0 torchvision==0.18.0 \
	  --index-url https://download.pytorch.org/whl/cu121
	@echo "✅ GPU install done. Verify with: python -c 'import torch; print(torch.cuda.get_device_name(0))'"

test:             ## Lancer les tests
	poetry run pytest

lint:             ## Vérifier le code
	poetry run ruff check src/ tests/

format:           ## Formater le code
	poetry run ruff format src/ tests/

docs:             ## Lancer la doc en local
	poetry run mkdocs serve

push-models:      ## Uploader les modèles entraînés sur HF Hub
	poetry run python scripts/push_models_to_hub.py
