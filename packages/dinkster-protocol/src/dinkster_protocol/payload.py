"""Canonical RPC-clean JSON payload validation.

One definition of "clean payload" shared by the durable event envelopes in
this package and the server's journal substrate, so what an appender may
write and what an envelope may carry can never drift apart. The canonical
encoding (sorted keys, no whitespace, no NaN/Infinity) is the payload's
identity everywhere it is persisted or digested.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import cast

__all__ = ["canonical_json_payload"]


def _require_rpc_clean(value: object, path: str) -> None:
    """Exact-type JSON tree check. Rejects everything json.dumps would
    silently alias into a different canonical value: bools/ints confused by
    equality, non-string object keys stringified, tuples flattened to
    arrays, and nonfinite floats emitted as nonstandard tokens."""
    if value is None or type(value) is bool or type(value) is str:
        return
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError(f"{path} integer is outside the signed 64-bit range")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} float must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(cast("list[object]", value)):
            _require_rpc_clean(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast("dict[object, object]", value).items():
            if type(key) is not str:
                raise ValueError(f"{path} object keys must be strings")
            _require_rpc_clean(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} must be RPC-clean JSON (None, bool, int, float, str, list, dict)")


def canonical_json_payload(payload: object, *, max_bytes: int, description: str) -> str:
    """Validate a string-keyed RPC-clean JSON object and return its
    canonical encoding, refusing anything larger than ``max_bytes``."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a string-keyed mapping")
    tree = dict(cast("Mapping[object, object]", payload))
    if any(type(key) is not str for key in tree):
        raise ValueError(f"{description} must be a string-keyed mapping")
    _require_rpc_clean(tree, description)
    canonical = json.dumps(tree, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(canonical.encode("utf-8")) > max_bytes:
        raise ValueError(
            f"{description} exceeds {max_bytes} canonical-JSON bytes;"
            " reference artifacts instead of inlining them"
        )
    return canonical
