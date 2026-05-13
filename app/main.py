from fastapi import FastAPI

from app.core.config import settings
from app.routers import health, hello

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
)

app.include_router(hello.router)
app.include_router(health.router)
