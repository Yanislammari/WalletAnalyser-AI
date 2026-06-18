from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.service.get_finance_data import fetch_data_for_ai
from app.train_prod_ai import create_prod_model

router = APIRouter()

is_fetching = False
is_training = False

@router.get("/create-prod-model")
async def fetch_data_for_model(background_tasks: BackgroundTasks) -> dict[str, str]:
    global is_fetching
    if is_fetching:
        raise HTTPException(status_code=409, detail="Fetch already running")
    
    async def run():
        global is_fetching
        is_fetching = True
        try:
            await fetch_data_for_ai()
            await create_prod_model()
        finally:
            is_fetching = False

    background_tasks.add_task(run)
    return {"message": "Data fetch started"}

@router.get("/train-model")
async def train_model(background_tasks: BackgroundTasks) -> dict[str, str]:
    global is_training
    if is_training:
        raise HTTPException(status_code=409, detail="Training already running")

    async def run():
        global is_training
        is_training = True
        try:
            await create_prod_model()
        finally:
            is_training = False

    background_tasks.add_task(run)
    return {"message": "Training started"}