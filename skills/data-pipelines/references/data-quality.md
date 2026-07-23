# Data Quality Validation

Two-layer validation for datasets: per-record Pydantic checks plus aggregate suite validation with Great Expectations.

## The Two Layers

Two complementary layers: Pydantic validates each record at construction time;
Great Expectations validates the dataset in aggregate — distributions, uniqueness,
ranges — and emits a durable report artifact.

## Great Expectations Suite

```python
"""Great Expectations suite for an image dataset manifest."""

import great_expectations as gx
from loguru import logger

COLUMNS = ["file_hash", "relative_path", "width", "height", "channels", "label", "group_id"]


def create_image_dataset_expectations(context: gx.DataContext) -> None:
    """Define the expectation suite for an image classification manifest."""
    suite = context.add_expectation_suite("image_classification_suite")
    gxe = gx.expectations
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(column_set=COLUMNS),
        *(gxe.ExpectColumnValuesToNotBeNull(column=c) for c in ("file_hash", "label")),
        *(gxe.ExpectColumnValuesToBeBetween(column=c, min_value=32, max_value=8192)
          for c in ("width", "height")),
        # Duplicate hashes mean duplicate images — a cross-split leakage vector.
        gxe.ExpectColumnValuesToBeUnique(column="file_hash"),
    ]
    for expectation in expectations:
        suite.add_expectation(expectation)
    logger.info("Created suite with {} expectations", len(expectations))
```

## In-Pipeline Gate

Mirror the cheapest checks as an in-pipeline gate — null counts per column,
`df["file_hash"].is_duplicated().sum()`, per-class counts from
`df.group_by("label").len()`, and a filter for undersized images — returning a
frozen Pydantic `QualityReport` and raising when it fails. Run the gate after
extraction *and* after splitting: a healthy manifest still yields empty or
single-class splits when ratios or group sizes are wrong.
