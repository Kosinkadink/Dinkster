"""Asset identity: a canonical content digest (DESIGN 3.12).

Identity is a property of bytes - it survives renames, moves, duplicate
copies, and machine hops. The canonical form is ``blake3:<64 hex>``,
deliberately the same form ComfyUI's asset database uses, so digests
interoperate with hashes the existing ecosystem already computed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blake3 import blake3

DIGEST_PREFIX = "blake3:"
_DIGEST_RE = re.compile(r"^blake3:[0-9a-f]{64}$")

CHUNK_SIZE = 8 * 1024 * 1024
"""Hash files in 8 MiB chunks (matches ComfyUI's scanner) - bounded memory
regardless of file size."""


class AssetError(Exception):
    """An asset could not be identified, cataloged, or materialized."""


def is_digest(text: str) -> bool:
    return _DIGEST_RE.match(text) is not None


def require_digest(text: str) -> str:
    if not is_digest(text):
        raise AssetError(f"not a canonical asset digest (expected 'blake3:<64 hex>'): {text!r}")
    return text


def _blake3() -> type[blake3]:
    """Lazy: hashing happens where Dinkster proper runs (blake3 installed);
    workers that only *resolve* digests - compat children riding a foreign
    interpreter via PYTHONPATH - never need it."""
    try:
        from blake3 import blake3 as hasher
    except ImportError:  # pragma: no cover - environment-specific
        raise AssetError(
            "computing asset digests requires the 'blake3' package, which "
            "is not installed in this interpreter; compute digests where "
            "Dinkster is installed (workers resolve pre-indexed digests only)"
        ) from None
    return hasher


def new_hasher() -> blake3:
    """A fresh incremental hasher for streaming digest computation (vault
    ingest, resumable scans). Same lazy-import posture as digest_file."""
    return _blake3()()


def digest_bytes(data: bytes) -> str:
    return DIGEST_PREFIX + _blake3()(data).hexdigest()


def digest_file(path: Path) -> str:
    hasher = _blake3()()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            hasher.update(chunk)
    return DIGEST_PREFIX + hasher.hexdigest()
