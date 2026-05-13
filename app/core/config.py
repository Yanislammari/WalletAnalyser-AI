from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "WalletAnalyser AI"
    version: str = "0.1.0"
    port: int = 8080

    class Config:
        env_file = ".env"


settings = Settings()
