"""Configuration model for ${project_name}.

Configuration is a frozen Pydantic V2 model so it can be validated once at the
boundary and then passed around without defensive copying. ``extra="forbid"``
turns a typo in an environment variable or a config file into an immediate,
readable error instead of a silently ignored setting.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ENV_PREFIX", "LibraryConfig", "LogLevel"]

ENV_PREFIX: Final[str] = "${package_name}_".upper()
"""Prefix for environment variables read by :meth:`LibraryConfig.from_env`."""

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
"""Loguru severity names accepted by :attr:`LibraryConfig.log_level`."""


class LibraryConfig(BaseModel):
    """Runtime settings for ${project_name}.

    Attributes:
        log_level: Minimum Loguru severity an embedding application should emit.
        strict: Fail fast on recoverable problems instead of warning and
            continuing. Libraries should default to strict and let the caller
            opt out.
        max_items: Upper bound on how many items a single operation will
            process, used to keep pathological inputs from exhausting memory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    log_level: LogLevel = Field(
        default="INFO",
        description="Minimum Loguru severity to emit.",
    )
    strict: bool = Field(
        default=True,
        description="Raise on recoverable problems instead of warning.",
    )
    max_items: int = Field(
        default=1024,
        ge=1,
        description="Maximum number of items processed by a single operation.",
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LibraryConfig:
        """Build a config from environment variables.

        Each field maps to ``ENV_PREFIX + FIELD_NAME.upper()`` — for example
        ``max_items`` reads from the ``MAX_ITEMS`` variable under
        :data:`ENV_PREFIX`. Unset variables fall back to the field default.

        Args:
            env: Mapping to read from. Defaults to :data:`os.environ`.

        Returns:
            A validated, frozen config instance.

        Raises:
            pydantic.ValidationError: If a present variable holds a value that
                does not coerce to the field's type.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        values = {
            field: source[f"{ENV_PREFIX}{field.upper()}"]
            for field in cls.model_fields
            if f"{ENV_PREFIX}{field.upper()}" in source
        }
        return cls.model_validate(values)

    def env_var_names(self) -> dict[str, str]:
        """Return a mapping of field name to the environment variable it reads."""
        return {field: f"{ENV_PREFIX}{field.upper()}" for field in type(self).model_fields}
