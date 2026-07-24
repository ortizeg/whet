# Large-Scale Processing

Parallel chunked processing with error isolation, and streaming reads for datasets larger than RAM.

## Chunked Parallel Processing

Chunked parallel processing with per-item error isolation and progress logging;
streaming reads for datasets larger than RAM.

```python
"""Parallel chunked processing and streaming reads."""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def process_in_chunks(
    paths: list[Path], fn: Callable[[Path], Any], workers: int = 8, chunk: int = 1000
) -> list[Any]:
    """Process files in parallel chunks; one failed item never kills the run."""
    results: list[Any] = []
    total = len(paths)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0, total, chunk):
            futures = {executor.submit(fn, p): p for p in paths[start : start + chunk]}
            for future in as_completed(futures):
                try:
                    results.append(future.result(timeout=300))
                except Exception:
                    logger.exception("Failed to process: {}", futures[future])
            done = min(start + chunk, total)
            logger.info("Progress: {}/{} ({:.1f}%)", done, total, done / total * 100)
    logger.info("Processed {}/{} files successfully", len(results), total)
    return results
```

## Streaming Reads

For datasets larger than RAM, never call `pl.read_parquet`. Scan lazily and pull
fixed-size slices: `reader = pl.scan_parquet(path)`, then
`reader.slice(offset, batch).collect()` per batch, with the row count from
`reader.select(pl.len()).collect().item()`.
