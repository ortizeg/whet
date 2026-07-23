# Data Engineer Agent

The Data Engineer Agent guides data pipeline architecture, data quality, and storage decisions for ML/CV projects.

**Agent directory:** `agents/data-engineer/`

## Purpose

This agent provides expert advice on building data pipelines for ML projects. It covers ETL/ELT patterns, data validation with Pydantic, dataset versioning with DVC, storage format selection, data augmentation pipeline design, and data quality monitoring.

## Strictness Level

**ADVISORY** — This agent guides decisions but does not block.

## When to Use

- Designing data pipelines for training data preparation
- Choosing storage formats (Parquet, Arrow, LMDB, TFRecord)
- Implementing data validation and quality checks
- Planning dataset versioning and lineage tracking
- Designing data augmentation strategies
- Handling schema evolution and migration

## Example Session

```
You: "I need to build a data pipeline for image classification with 100K images"

Data Engineer: "I recommend this approach:
1. Store raw images in S3/GCS, metadata in Parquet
2. DVC for dataset versioning with remote storage
3. Pydantic models for metadata validation
4. WebDataset or LMDB for training-time access
5. Great Expectations for automated quality checks"
```

## Related Skills

- `dvc` — Data versioning patterns
- `pydantic` — Data validation models
- `testing` — Data pipeline testing strategies

## Full Reference

See [`agents/data-engineer/SKILL.md`](https://github.com/ortizeg/whet/blob/main/agents/data-engineer/SKILL.md) for complete patterns.
