"""Minimal end-to-end example for ${project_name}.

Run with::

    python examples/basic_usage.py
"""

from __future__ import annotations

import sys

from loguru import logger

from ${package_name} import LibraryConfig, Registry, RegistryError


def main() -> int:
    """Build a registry, resolve config, and exercise both."""
    # An application — unlike the library — is expected to own its log sinks.
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{level: <8} | {message}")

    config = LibraryConfig.from_env()
    logger.info("Resolved configuration: {}", config.model_dump())

    encoders: Registry[str] = Registry("encoders")

    @encoders.register("upper")
    def upper_encoder(text: str) -> str:
        return text.upper()

    @encoders.register("reverse")
    def reverse_encoder(text: str) -> str:
        return text[::-1]

    logger.info("Registered encoders: {}", encoders.names())

    for name in encoders:
        logger.info("{} -> {}", name, encoders.create(name, "hello world"))

    try:
        encoders.create("rot13", "hello world")
    except RegistryError as exc:
        logger.warning("Expected failure for an unregistered key: {}", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
