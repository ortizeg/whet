"""Generate a tiny synthetic raw dataset so the pipeline is runnable on day one.

Writes real (if very small) PNG files laid out as ``<root>/<label>/<group>/<frame>.png``
using nothing but the standard library — no Pillow, no download, no GPU.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

from loguru import logger

DEFAULT_LABELS: tuple[str, ...] = ("goal", "no_goal")


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Encode a solid-colour RGB PNG. Small, valid, and decodable by any reader."""
    scanline = b"\x00" + bytes(rgb) * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanline * height, 9)),
            _png_chunk(b"IEND", b""),
        ]
    )


def _colour_for(key: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(key.encode()).digest()
    return (digest[0], digest[1], digest[2])


def generate_sample_dataset(
    root: Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    groups_per_label: int = 10,
    frames_per_group: int = 5,
    size: int = 8,
) -> int:
    """Write a synthetic clip-structured dataset and return the file count.

    Each ``group`` stands in for a video clip: its frames are correlated by
    construction, which is exactly the leakage the group-aware splitter prevents.
    """
    written = 0
    for label in labels:
        for group_index in range(groups_per_label):
            group_id = f"{label}_clip_{group_index:03d}"
            group_dir = root / label / group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in range(frames_per_group):
                path = group_dir / f"frame_{frame_index:04d}.png"
                path.write_bytes(encode_png(size, size, _colour_for(str(path.relative_to(root)))))
                written += 1
    logger.info(
        "Sample dataset: {} file(s) under {} ({} labels x {} groups x {} frames)",
        written,
        root,
        len(labels),
        groups_per_label,
        frames_per_group,
    )
    return written
