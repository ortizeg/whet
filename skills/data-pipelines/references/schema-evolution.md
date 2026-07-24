# Schema Evolution and Migration

Versioning dataset schemas as explicit Pydantic models with a registered migration chain.

## Principle

Never mutate a dataset schema in place. Declare each version as its own model,
register it, and provide an explicit migration function per hop.

## Versioned Schemas

```python
"""Versioned dataset schemas with an explicit migration chain."""

from pydantic import BaseModel, Field


class SchemaV1(BaseModel):
    """Original — path and label only."""
    image_path: str
    label: str


class SchemaV2(SchemaV1):
    """V2 — adds dimensions and content hash."""
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    file_hash: str


class SchemaV3(SchemaV2):
    """V3 — adds split assignment and quality score."""
    split: str = Field(pattern=r"^(train|val|test)$")
    quality_score: float = Field(ge=0.0, le=1.0, default=1.0)


def migrate_v2_to_v3(record: SchemaV2, split: str, quality_score: float = 1.0) -> SchemaV3:
    """Enrich a V2 record with its split assignment and quality score."""
    return SchemaV3(**record.model_dump(), split=split, quality_score=quality_score)


# Ordered registry: reading version N and targeting M applies hops N..M in order.
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "1.0.0": SchemaV1,
    "2.0.0": SchemaV2,
    "3.0.0": SchemaV3,
}
MIGRATIONS = {("2.0.0", "3.0.0"): migrate_v2_to_v3}
```

## When Not to Subclass

Subclassing keeps additive versions short, but write a standalone model whenever a
field changes meaning or type — inheritance must never hide a breaking change.
Store `schema_version` alongside the data (Parquet key-value metadata or a sidecar
`dataset.json`) so readers dispatch to the right model instead of guessing.
