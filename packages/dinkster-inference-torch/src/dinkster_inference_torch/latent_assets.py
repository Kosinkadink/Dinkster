"""Native latent safetensors serialization and ComfyUI format materialization."""

from __future__ import annotations

import importlib
import json
import struct
import tempfile
from typing import Any, BinaryIO, cast

from dinkster_assets import (
    LATENT_SCHEMA_KEY,
    MAX_LATENT_DATA_BYTES,
    MAX_LATENT_HEADER_BYTES,
    MAX_LATENT_SCHEMA_BYTES,
    MAX_LATENT_STREAMS,
    MAX_VAE_HINT_BYTES,
    AssetError,
    parse_latent_asset,
    valid_vae_hint,
)
from dinkster_inference import MultiStreamLatent

MAX_WORKFLOW_BYTES = 2 * 1024 * 1024
MAX_PROMPT_BYTES = 1024 * 1024

_TORCH_TO_SAFE = {
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}
_SAFE_TO_TORCH = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}


def _json_string(value: object, *, name: str, limit: int) -> str:
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise AssetError(f"latent {name} metadata is not JSON serializable") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise AssetError(f"latent {name} metadata exceeds {limit} bytes")
    return encoded


def _tensor_facts(tensor: object) -> tuple[str, tuple[int, ...], int]:
    dtype = _TORCH_TO_SAFE.get(str(getattr(tensor, "dtype", None)))
    shape_value = getattr(tensor, "shape", None)
    if dtype is None or shape_value is None:
        raise AssetError("latent samples must be a supported floating-point tensor")
    shape = tuple(int(dimension) for dimension in shape_value)
    if not 1 <= len(shape) <= 8 or any(dimension <= 0 for dimension in shape):
        raise AssetError("latent tensor shape must have 1 through 8 positive dimensions")
    elements = 1
    for dimension in shape:
        elements *= dimension
        if elements > MAX_LATENT_DATA_BYTES:
            raise AssetError("latent tensor shape exceeds the data limit")
    item_size = 2 if dtype in {"F16", "BF16"} else 4 if dtype == "F32" else 8
    return dtype, shape, elements * item_size


def serialize_native_latent(
    samples: object,
    *,
    snapshot: Any | None = None,
    vae_hint: str | None = None,
) -> BinaryIO:
    """Serialize one native latent into a seekable disk-backed spool."""
    if type(samples) is MultiStreamLatent:
        multi = cast("MultiStreamLatent[object]", samples)
        if len(multi.streams) > MAX_LATENT_STREAMS:
            raise AssetError("native latent supports at most 64 streams")
        values = tuple((stream.role, stream.payload) for stream in multi.streams)
        structure = "multi"
    else:
        values = ((None, samples),)
        structure = "single"

    rows: list[dict[str, object]] = []
    facts: list[tuple[str, object, int]] = []
    offset = 0
    for index, (role, tensor) in enumerate(values):
        dtype, shape, byte_length = _tensor_facts(tensor)
        if offset + byte_length > MAX_LATENT_DATA_BYTES:
            raise AssetError("latent tensor data exceeds 1 GiB")
        name = "dinkster_samples" if structure == "single" else f"dinkster_stream_{index:04d}"
        row: dict[str, object] = {"tensor": name, "dtype": dtype, "shape": list(shape)}
        if structure == "multi":
            assert role is not None
            if len(role.encode("utf-8")) > 128:
                raise AssetError("latent stream role exceeds 128 UTF-8 bytes")
            row = {"order": index, "role": role, **row}
        rows.append(row)
        facts.append((name, tensor, byte_length))
        offset += byte_length

    schema = json.dumps(
        {"format": "dinkster.latent", "version": 1, "structure": structure, "streams": rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(schema.encode("utf-8")) > MAX_LATENT_SCHEMA_BYTES:
        raise AssetError("Dinkster latent schema exceeds 64 KiB")
    metadata: dict[str, str] = {LATENT_SCHEMA_KEY: schema}
    if snapshot is not None:
        metadata["prompt"] = _json_string(snapshot.prompt, name="prompt", limit=MAX_PROMPT_BYTES)
        workflow = (
            None if snapshot.extra_pnginfo is None else snapshot.extra_pnginfo.get("workflow")
        )
        if workflow is not None:
            metadata["workflow"] = _json_string(workflow, name="workflow", limit=MAX_WORKFLOW_BYTES)
    if vae_hint is not None and len(vae_hint.encode("utf-8")) <= MAX_VAE_HINT_BYTES:
        metadata["dinkster_vae_hint"] = vae_hint

    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for row, (name, _tensor, byte_length) in zip(rows, facts, strict=True):
        header[name] = {
            "dtype": row["dtype"],
            "shape": row["shape"],
            "data_offsets": [offset, offset + byte_length],
        }
        offset += byte_length
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    if len(encoded) > MAX_LATENT_HEADER_BYTES:
        raise AssetError("latent safetensors header exceeds 4 MiB")

    spool = tempfile.TemporaryFile(mode="w+b")
    try:
        spool.write(struct.pack("<Q", len(encoded)))
        spool.write(encoded)
        torch = cast("Any", importlib.import_module("torch"))
        for _name, tensor, _byte_length in facts:
            cpu = cast("Any", tensor).detach().to(device="cpu").contiguous()
            raw = cpu.view(torch.uint8).numpy()
            spool.write(memoryview(raw))
        spool.seek(0)
        return cast("BinaryIO", spool)
    except BaseException:
        spool.close()
        raise


def load_latent(source: BinaryIO, torch: Any) -> tuple[object, str]:
    """Validate first, then materialize one CPU tensor at a time."""
    descriptor = parse_latent_asset(source)
    tensors: list[object] = []
    for tensor in descriptor.tensors:
        source.seek(tensor.data_offset)
        body = bytearray(tensor.byte_length)
        if cast("Any", source).readinto(body) != tensor.byte_length:
            raise AssetError("latent tensor body was truncated after validation")
        dtype = getattr(torch, _SAFE_TO_TORCH[tensor.dtype])
        value = torch.frombuffer(body, dtype=dtype).reshape(tensor.shape)
        if descriptor.profile == "comfyui-single":
            value = value.float()
            if descriptor.legacy_scale:
                value = value * (1 / 0.18215)
        tensors.append(value)
    samples: object
    if descriptor.profile == "dinkster-v1" and descriptor.tensors[0].role is not None:
        samples = MultiStreamLatent[object].from_pairs(
            (cast(str, item.role), value)
            for item, value in zip(descriptor.tensors, tensors, strict=True)
        )
    else:
        samples = tensors[0]
    return samples, valid_vae_hint(descriptor)
