"""Command-line entry point: ``python -m ${package_name} <command>``.

All human-facing output goes through Loguru (never ``print``), so the CLI and
the library share one log configuration and one place to change formatting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from ${package_name}.download import (
    ChecksumMismatchError,
    download_weights,
    sha256_file,
)
from ${package_name}.model_card import ModelCard
from ${package_name}.registry import (
    ModelRegistry,
    RegistryError,
    load_registry,
)


def configure_logging(level: str = "INFO") -> None:
    """Install a single, message-only stderr sink."""
    logger.remove()
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")


def _describe(card: ModelCard) -> str:
    metrics = ", ".join(
        f"{name}={value:g}"
        for evaluation in card.evaluations
        for name, value in evaluation.metrics.items()
    )
    suffix = f" | {metrics}" if metrics else ""
    return f"{card.key:<28} {card.task:<14} {card.architecture:<18} [{card.status}]{suffix}"


def _cmd_list(registry: ModelRegistry, args: argparse.Namespace) -> int:
    cards = registry.select(task=args.task, tag=args.tag, status=args.status)
    if not cards:
        logger.warning("No models matched the given filters.")
        return 1
    for card in cards:
        logger.info(_describe(card))
    return 0


def _cmd_show(registry: ModelRegistry, args: argparse.Namespace) -> int:
    card = registry.get(args.name, args.version)
    logger.info("\n{}", card.to_yaml().rstrip())
    return 0


def _cmd_download(registry: ModelRegistry, args: argparse.Namespace) -> int:
    card = registry.get(args.name, args.version)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    try:
        path = download_weights(card, cache_dir=cache_dir, force=args.force)
    except ChecksumMismatchError as exc:
        logger.error("Refusing to use {}: {}", card.key, exc)
        return 2
    logger.info("{}", path)
    return 0


def _cmd_verify(registry: ModelRegistry, args: argparse.Namespace) -> int:
    card = registry.get(args.name, args.version)
    path = Path(args.file)
    if not path.is_file():
        logger.error("No such file: {}", path)
        return 2
    actual = sha256_file(path)
    if actual != card.weights.sha256:
        logger.error("{}: expected {}, got {}", path, card.weights.sha256, actual)
        return 2
    logger.info("{}: SHA-256 matches {}", path, card.key)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``${package_name}`` argument parser."""
    parser = argparse.ArgumentParser(prog="${package_name}", description="${description}")
    parser.add_argument("--registry", default=None, help="Registry directory (default: registry/)")
    parser.add_argument("--log-level", default="INFO", help="Loguru level, e.g. DEBUG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="List registered models")
    listing.add_argument("--task", default=None, help="Filter by task")
    listing.add_argument("--tag", default=None, help="Filter by tag")
    listing.add_argument("--status", default=None, help="Filter by status")
    listing.set_defaults(handler=_cmd_list)

    show = subparsers.add_parser("show", help="Print one model card as YAML")
    show.add_argument("name")
    show.add_argument("--version", default=None)
    show.set_defaults(handler=_cmd_show)

    download = subparsers.add_parser("download", help="Download and verify weights")
    download.add_argument("name")
    download.add_argument("--version", default=None)
    download.add_argument("--cache-dir", default=None)
    download.add_argument("--force", action="store_true", help="Ignore any cached copy")
    download.set_defaults(handler=_cmd_download)

    verify = subparsers.add_parser("verify", help="Check a local file against a model card")
    verify.add_argument("name")
    verify.add_argument("file")
    verify.add_argument("--version", default=None)
    verify.set_defaults(handler=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        registry = load_registry(Path(args.registry) if args.registry else None)
    except RegistryError as exc:
        logger.error("{}", exc)
        return 2
    try:
        exit_code: int = args.handler(registry, args)
    except RegistryError as exc:
        logger.error("{}", exc)
        return 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
