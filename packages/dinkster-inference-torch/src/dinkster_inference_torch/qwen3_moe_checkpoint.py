"""Strict streamed loading for sharded Qwen3 MoE checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from dinkster_inference import detect_qwen3_moe_config
from dinkster_inference.sources import SafetensorsSource, load_safetensors_header

from .module_residency import ModuleStateStore
from .qwen3_moe import Qwen3MoeForCausalLM
from .sources import load_tensors_from_file


class Qwen3MoeCheckpointError(ValueError):
    """The shard set cannot form one exact Qwen3 MoE checkpoint."""


def _reserve_cuda_model_bytes(device: torch.device, size: int) -> None:
    if device.type != "cuda":
        return
    reservation = torch.empty(size, dtype=torch.uint8, device=device)
    del reservation


def load_qwen3_moe_checkpoint(
    shards: Sequence[Path],
    *,
    device: torch.device | str,
) -> Qwen3MoeForCausalLM:
    """Load one exact Qwen3-30B-A3B shard set onto ``device``.

    One shard is mapped and transferred at a time so a fully resident CUDA
    load does not materialize the full checkpoint in host memory first.
    """

    paths = tuple(shards)
    if not paths:
        raise ValueError("Qwen3 MoE checkpoint requires at least one shard")
    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("Qwen3 MoE checkpoint loading requires a materialized target device")

    sources: list[SafetensorsSource] = []
    geometries = {}
    for path in paths:
        source = load_safetensors_header(path)
        for key in source.keys():
            if key in geometries:
                raise Qwen3MoeCheckpointError(
                    f"Qwen3 MoE checkpoint key {key!r} appears in multiple shards"
                )
            geometries[key] = source.entry(key).geometry
        sources.append(source)

    config = detect_qwen3_moe_config(geometries)
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    store = ModuleStateStore(model)
    if store.keys() != geometries.keys():
        raise RuntimeError("Qwen3 MoE model state does not match the admitted checkpoint layout")

    model_bytes = sum(source.entry(key).nbytes for source in sources for key in source.keys())
    _reserve_cuda_model_bytes(target, model_bytes)
    for path, source in zip(paths, sources, strict=True):
        keys = sorted(source.keys(), key=lambda key: source.entry(key).offset)
        with path.open("rb") as handle:
            tensors = load_tensors_from_file(handle, source, keys)
            for key in keys:
                tensor = tensors.pop(key)
                moved = tensor.to(device=target)
                store[key] = moved
                del moved, tensor

    if any(tensor.device.type == "meta" for tensor in model.state_dict().values()):
        raise RuntimeError("Qwen3 MoE checkpoint loading left model state unmaterialized")
    return model.eval()


__all__ = [
    "Qwen3MoeCheckpointError",
    "load_qwen3_moe_checkpoint",
]
