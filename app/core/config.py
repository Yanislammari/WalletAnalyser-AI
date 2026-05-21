from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "WalletAnalyser AI"
    version: str = "0.1.0"
    database_url: str
    port: int

    class Config:
        env_file = ".env"

settings = Settings()
