
from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration centralisée de l'API UNIDET."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: SecretStr

    cohere_api_key: SecretStr
    cohere_embed_model: str = "embed-v4.0"
    cohere_embed_dimension: int = 1024

    groq_api_key: SecretStr
    groq_model: str

    retrieval_top_k: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()