"""Command-line entry point for the ${project_name} pipeline.

python -m ${package_name} sample --root data/raw
python -m ${package_name} scan --config conf/pipeline.toml
python -m ${package_name} validate --config conf/pipeline.toml
python -m ${package_name} run --config conf/pipeline.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ${package_name}.config import PipelineConfig
from ${package_name}.manifest import manifest_fingerprint
from ${package_name}.pipeline import run_pipeline
from ${package_name}.quality import validate_manifest
from ${package_name}.sample_data import DEFAULT_LABELS, generate_sample_dataset
from ${package_name}.splitting import DataLeakageError
from ${package_name}.stages import stage_ingest

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CONFIG = Path("conf/pipeline.toml")


def configure_logging(level: str) -> None:
    """One structured sink on stderr — never print()."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), backtrace=False, diagnose=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="${package_name}", description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="Loguru level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Write a synthetic raw dataset")
    sample.add_argument("--root", type=Path, default=Path("data/raw"))
    sample.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    sample.add_argument("--groups", type=int, default=4, help="Groups (clips) per label")
    sample.add_argument("--frames", type=int, default=5, help="Frames per group")

    for name, description in (
        ("scan", "Ingest the raw tree and report the manifest"),
        ("validate", "Ingest and run the data-quality gate"),
        ("run", "Run the full ingest -> validate -> split -> write pipeline"),
    ):
        sub = subparsers.add_parser(name, help=description)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    return parser


def _load_config(path: Path) -> PipelineConfig:
    if not path.exists():
        logger.warning("Config {} not found — falling back to defaults", path)
        return PipelineConfig()
    return PipelineConfig.from_toml(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the requested command, return a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        if args.command == "sample":
            labels = tuple(part.strip() for part in args.labels.split(",") if part.strip())
            generate_sample_dataset(
                root=args.root,
                labels=labels,
                groups_per_label=args.groups,
                frames_per_group=args.frames,
            )
            return 0

        config = _load_config(args.config)

        if args.command == "scan":
            manifest = stage_ingest(config.ingest)
            logger.info(
                "Manifest: {} records, {} groups, fingerprint {}",
                manifest.height,
                manifest["group_id"].n_unique() if manifest.height else 0,
                manifest_fingerprint(manifest)[:12],
            )
            return 0

        if args.command == "validate":
            manifest = stage_ingest(config.ingest)
            report = validate_manifest(manifest, config.quality)
            return 0 if report.ok else 1

        run_pipeline(config)
    except (DataLeakageError, FileNotFoundError, ValueError) as error:
        logger.error("Pipeline failed: {}", error)
        return 1
    return 0
