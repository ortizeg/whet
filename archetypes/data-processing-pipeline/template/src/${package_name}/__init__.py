"""${project_name} — ${description}

A leakage-free ETL pipeline for CV/ML datasets:
``ingest -> validate -> split -> write``.
"""

from ${package_name}.config import (
    IngestConfig,
    OutputConfig,
    PipelineConfig,
    QualityConfig,
    SplitConfig,
)
from ${package_name}.manifest import (
    MANIFEST_SCHEMA,
    SCHEMA_VERSION,
    ImageRecord,
    build_manifest,
    manifest_fingerprint,
    read_manifest,
    write_manifest,
)
from ${package_name}.pipeline import PipelineResult, run_pipeline
from ${package_name}.quality import (
    QualityIssue,
    QualityReport,
    Severity,
    validate_manifest,
)
from ${package_name}.splitting import (
    DataLeakageError,
    assert_no_group_leakage,
    group_aware_split,
    split_dataset,
    stratified_split,
)

__version__ = "0.1.0"

__all__ = [
    "MANIFEST_SCHEMA",
    "SCHEMA_VERSION",
    "DataLeakageError",
    "ImageRecord",
    "IngestConfig",
    "OutputConfig",
    "PipelineConfig",
    "PipelineResult",
    "QualityConfig",
    "QualityIssue",
    "QualityReport",
    "Severity",
    "SplitConfig",
    "__version__",
    "assert_no_group_leakage",
    "build_manifest",
    "group_aware_split",
    "manifest_fingerprint",
    "read_manifest",
    "run_pipeline",
    "split_dataset",
    "stratified_split",
    "validate_manifest",
    "write_manifest",
]
