"""Weight download and — the point of the whole module — SHA-256 verification.

No network is touched: sources are ``file://`` URLs or an injected fetcher.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ${package_name}.download import (
    ChecksumMismatchError,
    cached_path,
    default_fetcher,
    download_weights,
    sha256_file,
    verify_file,
)
from ${package_name}.model_card import ModelCard

MakeCard = Callable[..., ModelCard]


def test_sha256_file_matches_hashlib(local_weights: Path) -> None:
    assert sha256_file(local_weights) == hashlib.sha256(local_weights.read_bytes()).hexdigest()


def test_download_from_file_url(
    make_local_card: MakeCard, local_weights: Path, tmp_path: Path
) -> None:
    card = make_local_card()
    cache = tmp_path / "cache"

    path = download_weights(card, cache_dir=cache)

    assert path == cached_path(card, cache)
    assert path.read_bytes() == local_weights.read_bytes()
    assert sha256_file(path) == card.weights.sha256


def test_download_rejects_corrupted_file(make_local_card: MakeCard, tmp_path: Path) -> None:
    """The critical guarantee: bytes that do not match the card are refused."""
    wrong_digest = hashlib.sha256(b"these-are-not-the-bytes-you-are-looking-for").hexdigest()
    card = make_local_card(sha256=wrong_digest)
    cache = tmp_path / "cache"

    with pytest.raises(ChecksumMismatchError) as excinfo:
        download_weights(card, cache_dir=cache)

    assert excinfo.value.expected == wrong_digest
    assert excinfo.value.actual != wrong_digest

    destination = cached_path(card, cache)
    assert not destination.exists(), "corrupt weights must never land in the cache"
    assert list(destination.parent.glob("*.part")) == [], "partial download must be cleaned up"


def test_corrupt_cache_entry_is_replaced(
    make_local_card: MakeCard, local_weights: Path, tmp_path: Path
) -> None:
    card = make_local_card()
    cache = tmp_path / "cache"
    destination = cached_path(card, cache)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"tampered")

    path = download_weights(card, cache_dir=cache)

    assert path.read_bytes() == local_weights.read_bytes()


def test_cache_hit_skips_the_fetcher(make_local_card: MakeCard, tmp_path: Path) -> None:
    card = make_local_card()
    cache = tmp_path / "cache"
    download_weights(card, cache_dir=cache)

    def exploding_fetcher(url: str, /) -> Iterator[bytes]:
        msg = f"fetcher must not be called on a cache hit (url={url})"
        raise AssertionError(msg)

    assert download_weights(card, cache_dir=cache, fetcher=exploding_fetcher).exists()


def test_injected_fetcher_is_used(card_template: dict[str, Any], tmp_path: Path) -> None:
    payload = b"bytes-from-an-injected-fetcher"
    weights = {
        "url": "https://models.example.com/tiny-net.pth",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    card = ModelCard.model_validate({**card_template, "weights": weights})

    def fake_fetcher(url: str, /) -> Iterator[bytes]:
        assert url == card.weights.url
        yield payload[:10]
        yield payload[10:]

    path = download_weights(card, cache_dir=tmp_path / "cache", fetcher=fake_fetcher)
    assert path.read_bytes() == payload


def test_verify_file(local_weights: Path) -> None:
    assert verify_file(local_weights, sha256_file(local_weights))
    assert not verify_file(local_weights, "0" * 64)


def test_default_fetcher_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="no fetcher registered"):
        default_fetcher("s3://bucket/key.pth")


def test_cached_path_layout(make_local_card: MakeCard, tmp_path: Path) -> None:
    card = make_local_card()
    assert cached_path(card, tmp_path).relative_to(tmp_path) == Path(
        card.name, card.version, card.weights.filename
    )
