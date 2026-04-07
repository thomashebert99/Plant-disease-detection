FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./
COPY src ./src
COPY app ./app

CMD ["python", "-m", "http.server", "8000"]
