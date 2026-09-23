"""Compact array and byte storage with host-local residency accounting.

Bfloat16 uses a named uint16 field in numpy's portable array format, retaining
the exact bits even in interpreters without torch or a bfloat16 numpy dtype.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

from .model import RESOURCE_ID_META_KEY
from .resources import COST_META_KEY

BFLOAT16_FIELD = "bfloat16"


def byte_storage_meta(data: bytes) -> dict[str, object]:
    """Account for retained bytes, without interpreting or decoding their contents."""
    size = len(data)
    return {"storage_bytes": size, COST_META_KEY: {"ram": size}}


def encoded_storage_meta(metadata: Mapping[str, object], size: int) -> dict[str, object]:
    """Charge locally held codec bytes, never a sending host's device allocation."""
    local = dict(metadata)
    if ("storage_dtype" in local or "storage_bytes" in local) and RESOURCE_ID_META_KEY not in local:
        local[COST_META_KEY] = {"ram": size}
        if "storage_bytes" in local:
            local["storage_bytes"] = size
    return local


def storage_dtype(array: object) -> str:
    obj = cast(Any, array) if hasattr(array, "dtype") else storage_array(array)
    dtype = obj.dtype
    if (
        getattr(dtype, "names", None) == (BFLOAT16_FIELD,)
        and dtype.itemsize == 2
        and dtype[BFLOAT16_FIELD].kind == "u"
        and dtype[BFLOAT16_FIELD].itemsize == 2
    ):
        return "bf16"
    name = str(dtype).removeprefix("torch.")
    return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(name, name)


def array_storage_meta(array: object) -> dict[str, object]:
    """Describe host storage without copying a device tensor to the host."""
    obj = cast(Any, array)
    device = str(getattr(obj, "device", "cpu"))
    if device != "cpu":
        return {"storage_dtype": storage_dtype(array), COST_META_KEY: {}}
    backing = obj if hasattr(obj, "numel") or not hasattr(obj, "nbytes") else storage_array(obj)
    while getattr(backing, "base", None) is not None:
        backing = backing.base
    if hasattr(backing, "untyped_storage"):
        size = int(backing.untyped_storage().nbytes())
    elif hasattr(backing, "numel"):
        size = int(backing.numel()) * int(backing.element_size())
    elif isinstance(backing, (bytes, bytearray, memoryview)):
        size = memoryview(cast(Any, backing)).nbytes
    elif not hasattr(backing, "nbytes"):
        shape = cast("tuple[int, ...]", backing.shape)
        dtype_name = str(backing.dtype).removeprefix("torch.")
        itemsize = (
            2
            if dtype_name == "bfloat16"
            else importlib.import_module("numpy").dtype(dtype_name).itemsize
        )
        size = math.prod(shape) * itemsize
    else:
        size = int(backing.nbytes)
    return {
        "storage_dtype": storage_dtype(array),
        COST_META_KEY: {"ram": size},
    }


def storage_array(obj: object) -> Any:
    """Numpy storage view, preserving tensor dtype including bfloat16 bits."""
    np = importlib.import_module("numpy")
    if hasattr(obj, "detach"):
        tensor = cast(Any, obj).detach().cpu()
        if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
            torch = importlib.import_module("torch")
            return tensor.view(torch.uint16).numpy().view([(BFLOAT16_FIELD, "<u2")])
        obj = tensor.numpy()
    return np.asarray(obj)


def image_input(obj: object) -> object:
    """Normalize integer pixels and upcast only at a consumer boundary."""
    from .image_codec import copy_media_semantics

    kind = storage_dtype(obj)
    if hasattr(obj, "detach"):
        tensor = cast(Any, obj)
        if kind == "fp32":
            return tensor
        result = tensor.float()
    else:
        np = cast(Any, importlib.import_module("numpy"))
        array = np.asarray(obj)
        if kind == "fp32":
            return copy_media_semantics(obj, array)
        if kind == "bf16":
            result = (array[BFLOAT16_FIELD].astype(np.uint32) << 16).view(np.float32)
            return copy_media_semantics(obj, result)
        result = array.astype(np.float32)
    if kind == "uint8":
        result /= 255.0
    elif kind == "uint16":
        result /= 65535.0
    return copy_media_semantics(obj, result)


def audio_input(obj: object) -> object:
    """Normalize PCM16 while leaving produced float32 audio unchanged."""
    value = cast(Mapping[str, object], obj)
    waveform = value["waveform"]
    if storage_dtype(waveform) != "int16":
        return obj
    if hasattr(waveform, "detach"):
        normalized = cast(Any, waveform).float() / 32768.0
    else:
        np = cast(Any, importlib.import_module("numpy"))
        normalized = np.asarray(waveform).astype(np.float32) / 32768.0
    return {**value, "waveform": normalized}
