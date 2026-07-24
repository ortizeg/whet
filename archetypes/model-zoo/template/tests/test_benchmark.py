"""The benchmark harness works on any callable — no torch required."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ${package_name}.benchmark import (
    BenchmarkConfig,
    percentile,
    run_benchmark,
)


def test_benchmark_counts_iterations() -> None:
    calls = 0

    def step() -> None:
        nonlocal calls
        calls += 1

    config = BenchmarkConfig(warmup_iterations=3, iterations=10, batch_size=4)
    result = run_benchmark(step, name="noop", config=config)

    assert calls == 13, "warm-up runs happen but are not timed"
    assert result.iterations == 10
    assert result.batch_size == 4


def test_benchmark_result_is_coherent() -> None:
    result = run_benchmark(
        lambda: sum(range(1000)),
        name="sum",
        config=BenchmarkConfig(warmup_iterations=1, iterations=20),
    )

    assert result.min_latency_ms <= result.median_latency_ms <= result.max_latency_ms
    assert result.min_latency_ms <= result.p95_latency_ms <= result.max_latency_ms
    assert result.throughput_items_per_s > 0.0
    assert "sum" in result.summary()


def test_default_config_is_used() -> None:
    result = run_benchmark(lambda: None, name="default")
    assert result.iterations == BenchmarkConfig().iterations


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.0, 1.0), (0.5, 3.0), (0.95, 5.0), (1.0, 5.0)],
)
def test_percentile(fraction: float, expected: float) -> None:
    assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], fraction) == expected


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="fraction must be"):
        percentile([1.0], 1.5)


def test_config_validates() -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(iterations=0)
    with pytest.raises(ValidationError):
        BenchmarkConfig(warmup_iterations=-1)
