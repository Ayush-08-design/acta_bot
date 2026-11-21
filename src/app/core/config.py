from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    DB_URL: str
    POOL_SIZE: Annotated[int, Field(ge=5, le=100)] = 20


class AuthSettings(BaseSettings):
    SECRET_KEY: SecretStr
    ALGORITHM: SecretStr
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int


class ApiKeySettings(BaseSettings):
    TAVILY_API_KEY: SecretStr
    GROQ_API_KEY: SecretStr


class AppSettings(BaseSettings):
    # Configuration & Sources
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # Supports nested structure parsing
        env_ignore_empty=True,  # Empty string environment variables are ignored
        extra="ignore",  # Silently ignore arbitrary environment clutter
    )

    # General Environment & Context
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    AUTH: AuthSettings

    API: ApiKeySettings

    DB: DatabaseSettings

    # Custom Field Validator for Business Logic
    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def lowercase_environment(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else v


# Singleton Pattern with Caching
@lru_cache
def get_settings() -> AppSettings:
    """Loads and caches settings so parsing overhead is incurred exactly once."""
    return AppSettings()
