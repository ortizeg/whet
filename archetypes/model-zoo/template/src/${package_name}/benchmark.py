"""Latency and throughput benchmark harness.

The harness is framework-agnostic on purpose: it times an arbitrary zero-arg
callable, so the same code benchmarks a torch module, an ONNX Runtime session,
or a pure-Python stub. That keeps the benchmark importable — and testable —
without a deep-learning stack installed.

    result = run_benchmark(lambda: session.run(None, feeds), name="resnet50")
    logger.info(result.summary())
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from statistics import fmean, median

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid", frozen=True)


class BenchmarkConfig(BaseModel):
    """How many times to run the step, and how many items each run covers."""

    model_config = _STRICT

    warmup_iterations: int = Field(default=5, ge=0, description="Untimed warm-up runs")
    iterations: int = Field(default=50, gt=0, description="Timed runs")
    batch_size: int = Field(default=1, gt=0, description="Items processed per run")


class BenchmarkResult(BaseModel):
    """Timing summary for one benchmarked callable."""

    model_config = _STRICT

    name: str
    iterations: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    mean_latency_ms: float = Field(ge=0.0)
    median_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    min_latency_ms: float = Field(ge=0.0)
    max_latency_ms: float = Field(ge=0.0)
    throughput_items_per_s: float = Field(ge=0.0)

    def summary(self) -> str:
        """Return a single-line, human-readable summary."""
        return (
            f"{self.name}: mean {self.mean_latency_ms:.3f} ms | "
            f"p50 {self.median_latency_ms:.3f} ms | p95 {self.p95_latency_ms:.3f} ms | "
            f"{self.throughput_items_per_s:.1f} items/s "
            f"(n={self.iterations}, batch={self.batch_size})"
        )


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile of ``values`` (0.0 <= fraction <= 1.0)."""
    if not values:
        msg = "percentile requires at least one value"
        raise ValueError(msg)
    if not 0.0 <= fraction <= 1.0:
        msg = f"fraction must be in [0, 1], got {fraction}"
        raise ValueError(msg)
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def run_benchmark(
    step: Callable[[], object],
    *,
    name: str,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Time ``step`` repeatedly and summarize its latency and throughput."""
    settings = config if config is not None else BenchmarkConfig()

    for _ in range(settings.warmup_iterations):
        step()

    latencies_ms: list[float] = []
    for _ in range(settings.iterations):
        started = time.perf_counter()
        step()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    mean_ms = fmean(latencies_ms)
    throughput = (settings.batch_size * 1000.0 / mean_ms) if mean_ms > 0.0 else 0.0

    result = BenchmarkResult(
        name=name,
        iterations=settings.iterations,
        batch_size=settings.batch_size,
        mean_latency_ms=mean_ms,
        median_latency_ms=median(latencies_ms),
        p95_latency_ms=percentile(latencies_ms, 0.95),
        min_latency_ms=min(latencies_ms),
        max_latency_ms=max(latencies_ms),
        throughput_items_per_s=throughput,
    )
    logger.debug("Benchmark complete: {}", result.summary())
    return result
