FROM python:3.11.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/data/.huggingface \
    MONITORING_STORAGE_DIR=/data/plant-disease-detection/monitoring
ENV PIP_DEFAULT_TIMEOUT=120

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements-api-cpu.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-api-cpu.txt

COPY src ./src

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
