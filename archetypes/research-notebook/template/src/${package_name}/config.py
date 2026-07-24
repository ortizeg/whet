"""Experiment configuration and canonical paths for ${project_name}.

Notebooks must never hard-code a filesystem path or a magic hyperparameter. They
import :data:`PATHS` and construct an :class:`ExperimentConfig`, which is validated
by Pydantic V2 at construction time — so a typo fails immediately instead of three
cells later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ``src/<package>/config.py`` -> parents[2] is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    """Canonical project directories, all resolved from the project root.

    Every directory a notebook is allowed to read from or write to lives here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Field(default=PROJECT_ROOT, description="Project root directory")

    @property
    def data(self) -> Path:
        """Root of the data tree (never committed)."""
        return self.root / "data"

    @property
    def raw(self) -> Path:
        """Original, immutable inputs. Read-only by convention."""
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        """Derived data produced by a reproducible step."""
        return self.data / "processed"

    @property
    def outputs(self) -> Path:
        """Root of the generated-artifact tree (never committed)."""
        return self.root / "outputs"

    @property
    def figures(self) -> Path:
        """Saved figures. Notebooks save here rather than committing cell outputs."""
        return self.outputs / "figures"

    @property
    def reports(self) -> Path:
        """Generated reports, metric dumps, and exported tables."""
        return self.outputs / "reports"

    @property
    def notebooks(self) -> Path:
        """Notebook directory."""
        return self.root / "notebooks"

    def ensure(self) -> Self:
        """Create every managed directory if it does not already exist."""
        for directory in (self.raw, self.processed, self.figures, self.reports):
            directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured project directories under {}", self.root)
        return self


PATHS: Paths = Paths()
"""Process-wide path singleton — import this instead of building strings."""


class ExperimentConfig(BaseModel):
    """A single, fully specified experiment.

    Frozen on purpose: an experiment's parameters must not drift halfway through a
    notebook session. Use :meth:`variant` to derive a modified copy for an ablation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Slug identifying the run; used for output filenames",
    )
    seed: int = Field(default=42, ge=0, description="Seed for every RNG in the run")
    n_samples: int = Field(default=600, gt=0, le=1_000_000, description="Dataset size")
    n_features: int = Field(default=2, ge=2, le=512, description="Feature dimensionality")
    n_classes: int = Field(default=3, ge=2, le=100, description="Number of classes")
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0, description="Train split share")
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0, description="Validation share")
    stage: Literal["explore", "ablation", "final"] = Field(
        default="explore", description="How seriously the results should be taken"
    )
    notes: str = Field(default="", max_length=2000, description="Free-form research notes")
    tags: tuple[str, ...] = Field(default=(), description="Short labels for grouping runs")

    @model_validator(mode="after")
    def _splits_must_leave_room_for_test(self) -> Self:
        """Guard against a train/val split that leaves an empty test set."""
        if self.train_fraction + self.val_fraction >= 1.0:
            msg = (
                "train_fraction + val_fraction must be < 1.0 to leave a test split "
                f"(got {self.train_fraction} + {self.val_fraction})"
            )
            raise ValueError(msg)
        return self

    @property
    def test_fraction(self) -> float:
        """Whatever is left after train and validation."""
        return round(1.0 - self.train_fraction - self.val_fraction, 10)

    @property
    def slug(self) -> str:
        """Filename-safe identifier combining name and seed."""
        return f"{self.name}-seed{self.seed}"

    def figure_path(self, label: str) -> Path:
        """Return the canonical output path for a figure belonging to this run."""
        return PATHS.figures / f"{self.slug}-{label}.png"

    def variant(self, **overrides: object) -> ExperimentConfig:
        """Return a validated copy with ``overrides`` applied (for ablations)."""
        return ExperimentConfig.model_validate({**self.model_dump(), **overrides})

    def log_summary(self) -> None:
        """Emit the config to the log so the notebook's provenance is recoverable."""
        logger.info(
            "experiment={} stage={} seed={} n={} classes={} splits={:.2f}/{:.2f}/{:.2f}",
            self.name,
            self.stage,
            self.seed,
            self.n_samples,
            self.n_classes,
            self.train_fraction,
            self.val_fraction,
            self.test_fraction,
        )
