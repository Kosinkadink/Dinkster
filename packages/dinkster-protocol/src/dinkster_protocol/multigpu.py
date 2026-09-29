"""Run-scoped single-job multi-GPU execution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

SingleJobMultiGpuMode = Literal["auto", "guidance", "sequence", "window"]

_MODES = ("auto", "guidance", "sequence", "window")


@dataclass(frozen=True, slots=True)
class SingleJobMultiGpuConfig:
    """Ordered host CUDA lanes and execution mode selected for one run."""

    cuda_indices: tuple[int, ...]
    mode: SingleJobMultiGpuMode

    def __post_init__(self) -> None:
        if (
            type(self.cuda_indices) is not tuple
            or len(self.cuda_indices) < 2
            or any(type(index) is not int or index < 0 for index in self.cuda_indices)
            or len(set(self.cuda_indices)) != len(self.cuda_indices)
        ):
            raise ValueError("single-job CUDA ranks require at least two unique logical indices")
        if self.mode not in _MODES:
            raise ValueError("single-job multi-GPU mode is invalid")


@dataclass(frozen=True, slots=True)
class SingleJobMultiGpuExecution:
    """Rank-local distributed facts resolved from a run selection."""

    rank: int
    world_size: int
    mode: SingleJobMultiGpuMode

    def __post_init__(self) -> None:
        if (
            type(self.rank) is not int
            or type(self.world_size) is not int
            or self.world_size < 2
            or not 0 <= self.rank < self.world_size
        ):
            raise ValueError("single-job rank and world size are invalid")
        if self.mode not in _MODES:
            raise ValueError("single-job multi-GPU mode is invalid")


def single_job_multi_gpu_config_to_wire(config: SingleJobMultiGpuConfig) -> dict[str, object]:
    if type(config) is not SingleJobMultiGpuConfig:
        raise TypeError("single-job multi-GPU config must be exact")
    return {"cudaIndices": list(config.cuda_indices), "mode": config.mode}


def single_job_multi_gpu_config_from_wire(raw: object) -> SingleJobMultiGpuConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("single-job multi-GPU config must contain cudaIndices and mode")
    values = cast("Mapping[object, object]", raw)
    if set(values) != {"cudaIndices", "mode"}:
        raise ValueError("single-job multi-GPU config must contain cudaIndices and mode")
    indices = values["cudaIndices"]
    if not isinstance(indices, (list, tuple)):
        raise ValueError("single-job multi-GPU cudaIndices must be a list")
    return SingleJobMultiGpuConfig(
        tuple(cast("list[int] | tuple[int, ...]", indices)),
        cast("SingleJobMultiGpuMode", values["mode"]),
    )


def single_job_multi_gpu_execution_to_wire(
    execution: SingleJobMultiGpuExecution,
) -> dict[str, object]:
    if type(execution) is not SingleJobMultiGpuExecution:
        raise TypeError("single-job multi-GPU execution must be exact")
    return {
        "rank": execution.rank,
        "worldSize": execution.world_size,
        "mode": execution.mode,
    }


def single_job_multi_gpu_execution_from_wire(raw: object) -> SingleJobMultiGpuExecution:
    if not isinstance(raw, Mapping):
        raise ValueError("single-job multi-GPU execution must contain rank, worldSize, and mode")
    values = cast("Mapping[object, object]", raw)
    if set(values) != {"rank", "worldSize", "mode"}:
        raise ValueError("single-job multi-GPU execution must contain rank, worldSize, and mode")
    return SingleJobMultiGpuExecution(
        cast("int", values["rank"]),
        cast("int", values["worldSize"]),
        cast("SingleJobMultiGpuMode", values["mode"]),
    )


__all__ = [
    "SingleJobMultiGpuConfig",
    "SingleJobMultiGpuExecution",
    "SingleJobMultiGpuMode",
    "single_job_multi_gpu_config_from_wire",
    "single_job_multi_gpu_config_to_wire",
    "single_job_multi_gpu_execution_from_wire",
    "single_job_multi_gpu_execution_to_wire",
]
