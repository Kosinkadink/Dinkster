"""CPU checkpoint loading matching ComfyUI utils.py at 25dfc16f9ac0.

Safetensors use the shared native mapped reader. Legacy Torch files are always
weights-only loads, with the reference's inert training-metadata placeholders.
ComfyUI CLI mmap overrides are not native runtime policy.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from dinkster_inference.sources import load_safetensors_header_from_file

from .sources import load_tensors_from_file

_LEGACY_LOAD_LOCK = threading.RLock()


class ModelCheckpoint:
    pass


def scalar(*args: object, **kwargs: object) -> None:
    return None


def encode(*args: object, **kwargs: object) -> None:
    return None


# Torch 2.5 resolves callable metadata and does not support string alias tuples.
ModelCheckpoint.__module__ = "pytorch_lightning.callbacks.model_checkpoint"
scalar.__module__ = "numpy.core.multiarray"
encode.__module__ = "_codecs"


def load_checkpoint_with_metadata(path: Path) -> tuple[object, dict[str, str] | None]:
    """Load CPU state and safetensors metadata."""
    if str(path).lower().endswith((".safetensors", ".sft")):
        with path.open("rb") as file:
            source = load_safetensors_header_from_file(file, path=path)
            return load_tensors_from_file(file, source), dict(source.metadata()) or None

    from numpy import dtype
    from numpy.dtypes import Float64DType

    # Torch's safe-global contexts mutate one process-wide set without reference
    # counting. Serialize legacy loads and leave prior registrations untouched.
    with _LEGACY_LOAD_LOCK:
        existing = torch.serialization.get_safe_globals()
        allowed = [ModelCheckpoint, scalar, dtype, Float64DType, encode]
        with torch.serialization.safe_globals([item for item in allowed if item not in existing]):
            payload: Any = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    if "state_dict" in payload:
        return payload["state_dict"], None
    if len(payload) == 1:
        value = payload[next(iter(payload))]
        if isinstance(value, dict):
            return value, None
    return payload, None


def load_checkpoint(path: Path) -> object:
    """Load CPU state, unwrapping state_dict or a sole nested dictionary."""
    return load_checkpoint_with_metadata(path)[0]
