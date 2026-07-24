"""Command-line interface for ${project_name}.

The CLI is a thin diagnostic shell over the library: it reports the installed
version, the configuration resolved from the environment, and the contents of
:data:`${package_name}.component_registry`. Machine-readable payloads go to
stdout as JSON; human-readable diagnostics go to stderr through Loguru.

This module is the only place in the package that configures a Loguru sink, and
it does so inside :func:`main` rather than at import time, so importing the
library never hijacks an application's logging setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from loguru import logger

from ${package_name} import __version__
from ${package_name}.config import LibraryConfig
from ${package_name}.core import component_registry

__all__ = ["build_parser", "main", "run"]

PROG = "${project_slug}"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``${project_slug}`` command."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="${description}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level resolved from the environment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="Show version and resolved configuration.")
    subparsers.add_parser("components", help="List keys in the component registry.")
    return parser


def _configure_logging(level: str) -> None:
    """Install a stderr Loguru sink at ``level``, replacing any existing sinks."""
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level: <8} | {message}")


def _resolve_config(log_level: str | None) -> LibraryConfig:
    """Resolve config from the environment, applying an optional CLI override."""
    config = LibraryConfig.from_env()
    if log_level is None:
        return config
    return LibraryConfig.model_validate({**config.model_dump(), "log_level": log_level})


def _emit(payload: dict[str, Any]) -> None:
    """Write ``payload`` to stdout as indented JSON followed by a newline."""
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector to parse. Defaults to :data:`sys.argv` ``[1:]``.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a handled error.
    """
    args = build_parser().parse_args(argv)
    config = _resolve_config(args.log_level)
    _configure_logging(config.log_level)

    if args.command == "info":
        logger.debug("Resolved configuration from environment prefix.")
        _emit(
            {
                "package": "${package_name}",
                "version": __version__,
                "config": config.model_dump(),
                "env_vars": config.env_var_names(),
                "components": len(component_registry),
            }
        )
        return 0

    if args.command == "components":
        names = component_registry.names()
        if not names:
            logger.warning("No components registered in {!r}.", component_registry.name)
        _emit({"registry": component_registry.name, "components": names})
        return 0

    logger.error("Unknown command: {!r}", args.command)
    return 1


def run() -> None:
    """Console-script entry point declared in ``[project.scripts]``."""
    raise SystemExit(main())


if __name__ == "__main__":
    run()
