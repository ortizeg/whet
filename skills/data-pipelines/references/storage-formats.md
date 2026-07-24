# Storage Format Selection

Choosing between Parquet, LMDB, WebDataset, TFRecord, Arrow, and raw folders for CV/ML data.

## Decision Tree

```
Choosing a storage format?
├── Tabular metadata (labels, splits, paths, hashes)
│   └── Parquet ....... columnar, compressed, fast predicate filtering (Polars/DuckDB)
├── Streaming large image/video datasets
│   ├── WebDataset .... .tar shards; distributed/multi-node, S3-friendly, no random access
│   └── TFRecord ...... TensorFlow ecosystem, sequential reads
├── Random-access image datasets (single node, reshuffled every epoch)
│   └── LMDB .......... memory-mapped zero-copy reads, single-writer, bad over NFS
├── In-memory analytics / cross-language interchange
│   └── Arrow IPC ..... zero-copy, language-agnostic
└── Small datasets (< 1 GB) or active annotation churn
    └── Raw folders + Parquet manifest ..... simplest, debuggable, diffable
```

## Rules of Thumb

Raw files until random reads become the bottleneck; LMDB for
single-node random access; WebDataset once training spans nodes or lives in object
storage; Parquet for everything tabular, always. Write manifests with Hive-style
partitions (`split=train/data.parquet`, `compression="zstd"`,
`row_group_size=10_000`) so readers can skip whole splits.

Never use CSV for large datasets.

## LMDB Packing

```python
"""LMDB packing for fast random-access image reads."""

from pathlib import Path

import lmdb
from loguru import logger


def build_lmdb_dataset(image_paths: list[Path], out_path: Path, map_gb: int = 50) -> None:
    """Pack encoded image bytes into LMDB keyed by zero-padded index."""
    env = lmdb.open(str(out_path), map_size=map_gb * 1024**3)
    with env.begin(write=True) as txn:
        for idx, img_path in enumerate(image_paths):
            txn.put(f"{idx:08d}".encode(), img_path.read_bytes())
        txn.put(b"__len__", str(len(image_paths)).encode())
    env.close()
    logger.info("Built LMDB: {} images at {}", len(image_paths), out_path)
```
