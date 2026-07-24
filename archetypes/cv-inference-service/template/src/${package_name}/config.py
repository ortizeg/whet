"""Runtime settings for ${project_name}.

Every value is read from an environment variable (or a local ``.env`` file) and
validated at import time by Pydantic V2. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_path`` / ``model_name`` would otherwise collide with Pydantic's
        # reserved ``model_`` namespace.
        protected_namespaces=(),
    )

    app_host: str = Field(default="0.0.0.0", description="Bind address")  # noqa: S104
    app_port: int = Field(default=8000, ge=1, le=65535, description="Listen port")

    model_name: str = Field(default="default", description="Logical model name")
    model_path: Path = Field(
        default=Path("models/model.onnx"),
        description="Path to the ONNX model file; the service starts even if it is absent",
    )
    device: Literal["cpu", "cuda"] = Field(default="cpu", description="Inference device")

    input_width: int = Field(default=640, gt=0, description="Model input width in pixels")
    input_height: int = Field(default=640, gt=0, description="Model input height in pixels")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Default minimum detection confidence"
    )

    log_level: str = Field(default="INFO", description="Loguru log level")
    cors_origins: str = Field(default="*", description="Comma-separated allowed CORS origins")

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse ``cors_origins`` into the list FastAPI's CORS middleware expects."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
