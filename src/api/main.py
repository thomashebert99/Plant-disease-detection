"""Main FastAPI application entrypoint."""

from fastapi import FastAPI

from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router

app = FastAPI(title="Plant Disease Detection API", version="0.1.0")
app.include_router(health_router)
app.include_router(predict_router)
