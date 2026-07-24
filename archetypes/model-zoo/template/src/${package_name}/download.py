"""SHA-256-verified weight download with a local cache.

The contract is simple and non-negotiable: a weight file is only ever handed
back to the caller if its SHA-256 digest matches the value recorded in the
model card. A mismatch raises :class:`ChecksumMismatchError` and leaves no
partial file behind — corrupt or tampered weights never reach the cache.

Fetching is injectable. The default fetcher dispatches on URL scheme
(``http``/``https`` via httpx, ``file`` via the filesystem), but any callable
returning an iterator of byte chunks works, which keeps tests offline.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit
from urllib.request import url2pathname

import httpx
from loguru import logger

from ${package_name}.model_card import ModelCard

CHUNK_SIZE = 1 << 20
CACHE_DIR_ENV_VAR = "MODEL_WEIGHTS_CACHE"
DEFAULT_CACHE_DIR = Path(".cache") / "weights"
HTTP_TIMEOUT_S = 60.0


class ChecksumMismatchError(RuntimeError):
    """Raised when downloaded bytes do not match the model card's SHA-256."""

    def __init__(self, url: str, expected: str, actual: str) -> None:
        super().__init__(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")
        self.url = url
        self.expected = expected
        self.actual = actual


class Fetcher(Protocol):
    """Callable that streams the bytes at ``url`` in chunks."""

    def __call__(self, url: str, /) -> Iterator[bytes]: ...


def fetch_file(url: str) -> Iterator[bytes]:
    """Stream a ``file://`` URL from the local filesystem."""
    path = Path(url2pathname(urlsplit(url).path))
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            yield chunk


def fetch_http(url: str, *, timeout: float = HTTP_TIMEOUT_S) -> Iterator[bytes]:
    """Stream an ``http(s)://`` URL with httpx, following redirects."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        yield from response.iter_bytes(chunk_size=CHUNK_SIZE)


def default_fetcher(url: str, /) -> Iterator[bytes]:
    """Dispatch to the fetcher appropriate for the URL scheme."""
    scheme = urlsplit(url).scheme.lower()
    if scheme == "file":
        return fetch_file(url)
    if scheme in {"http", "https"}:
        return fetch_http(url)
    msg = f"no fetcher registered for URL scheme {scheme!r}"
    raise ValueError(msg)


def default_cache_dir() -> Path:
    """Return the weight cache directory, honouring ``MODEL_WEIGHTS_CACHE``."""
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    return Path(override) if override else DEFAULT_CACHE_DIR


def sha256_file(path: Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the lowercase hex SHA-256 digest of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> bool:
    """Return True when ``path`` hashes to ``expected_sha256``."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        logger.warning(
            "Checksum mismatch for {}: expected {}, got {}", path, expected_sha256, actual
        )
        return False
    return True


def cached_path(card: ModelCard, cache_dir: Path | None = None) -> Path:
    """Return the path a card's weights are (or would be) cached at."""
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / card.name / card.version / card.weights.filename


def download_weights(
    card: ModelCard,
    *,
    cache_dir: Path | None = None,
    fetcher: Fetcher | None = None,
    force: bool = False,
) -> Path:
    """Download ``card``'s weights into the cache and verify their SHA-256.

    Returns the path to the verified file. Raises :class:`ChecksumMismatchError`
    if the fetched bytes do not match the digest recorded in the card; in that
    case nothing is written to the cache.
    """
    destination = cached_path(card, cache_dir)
    fetch: Fetcher = fetcher if fetcher is not None else default_fetcher
    expected = card.weights.sha256

    if destination.exists() and not force:
        if verify_file(destination, expected):
            logger.debug("Cache hit for {} at {}", card.key, destination)
            return destination
        logger.warning("Cached weights for {} are corrupt; re-downloading", card.key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()

    logger.info("Downloading weights for {} from {}", card.key, card.weights.url)
    try:
        with partial.open("wb") as handle:
            for chunk in fetch(card.weights.url):
                digest.update(chunk)
                handle.write(chunk)

        actual = digest.hexdigest()
        if actual != expected:
            raise ChecksumMismatchError(card.weights.url, expected, actual)

        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    logger.success("Verified weights for {} at {}", card.key, destination)
    return destination
