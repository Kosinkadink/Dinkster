"""Vendored-data loading: gzipped package resources with pinned hashes.

Every byte blob under ``dinkster_inference/data`` carries a recorded
sha256 of its UNCOMPRESSED content in the module that consumes it;
loading verifies the pin so provenance is a guarantee, not a claim.
"""

from __future__ import annotations

import gzip
import hashlib
from importlib import resources


def read_vendored(name: str, expected_sha256: str) -> bytes:
    """Decompress ``dinkster_inference/data/<name>`` and verify its
    uncompressed sha256 against the recorded pin."""
    raw = gzip.decompress(resources.files("dinkster_inference.data").joinpath(name).read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"vendored data {name} hash mismatch: {digest} (expected {expected_sha256})"
        )
    return raw


__all__ = ["read_vendored"]
