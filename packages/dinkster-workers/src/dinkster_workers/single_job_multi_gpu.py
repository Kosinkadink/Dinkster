"""Parent-owned fixed selection contract for one job spanning CUDA ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SingleJobMultiGpuMode = Literal["auto", "guidance", "sequence", "window"]


@dataclass(frozen=True, slots=True)
class SingleJobMultiGpuConfig:
    """Ordered logical CUDA ranks fixed for the lifetime of a server."""

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
        if self.mode not in ("auto", "guidance", "sequence", "window"):
            raise ValueError("single-job multi-GPU mode is invalid")
