from fastapi import FastAPI
import pandas as pd
from pathlib import Path

from app.core.config import settings
from app.routers import finance_route, health

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
)

app.include_router(finance_route.router)
app.include_router(health.router)
