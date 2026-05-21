from fastapi import FastAPI
import pandas as pd
from pathlib import Path

from app.core.config import settings
from app.routers import finance_route, health
from app.service.ai_training_service import create_prod_model

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
)

df_clustered, hdb_model = create_prod_model()
# df_clustered = pd.read_csv(BASE_DIR / "data/metrics_cluster.csv")
# print(df_clustered["cluster"])

app.include_router(finance_route.router)
app.include_router(health.router)
