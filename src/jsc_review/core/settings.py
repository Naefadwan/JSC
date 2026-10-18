"""Environment-backed application settings with local-only defaults."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from JSC-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="JSC_",
        extra="ignore",
    )

    env: Literal["local", "test", "commission"] = "local"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings."""

    return Settings()
