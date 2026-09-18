"""Governed resident-model cache for line and edge preprocessors."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dinkster_api.v1 import ConsumerItem, FullReleaseResult, PressureSignal

if TYPE_CHECKING:
    import torch


@dataclass
class _Resident:
    model: torch.nn.Module
    bytes: int
    residency: str
    uses: int = 0
    parked: bool = False


class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._available = threading.Condition(self._lock)
        self._models: dict[str, _Resident] = {}

    @staticmethod
    def device() -> torch.device:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @contextmanager
    def use(
        self,
        key: str,
        factory: Callable[[], torch.nn.Module],
        *,
        device: torch.device | None = None,
    ) -> Iterator[torch.nn.Module]:
        with self._available:
            resident = self._models.get(key)
            while resident is not None and resident.parked:
                self._available.wait()
                resident = self._models.get(key)
            if resident is None:
                target = self.device() if device is None else device
                model = factory().eval().to(target)
                nbytes = sum(
                    value.numel() * value.element_size()
                    for value in (*model.parameters(), *model.buffers())
                )
                if target.type == "cuda":
                    import torch

                    index = (
                        target.index if target.index is not None else torch.cuda.current_device()
                    )
                    residency = f"vram:cuda:{index}"
                else:
                    residency = "ram"
                resident = _Resident(model=model, bytes=nbytes, residency=residency)
                self._models[key] = resident
            resident.uses += 1
        try:
            yield resident.model
        finally:
            with self._lock:
                resident.uses -= 1

    def footprint(self, device: str) -> int:
        with self._lock:
            return sum(item.bytes for item in self._models.values() if item.residency == device)

    def discard(self, key: str) -> int:
        with self._lock:
            resident = self._models.get(key)
            if resident is None or resident.uses or resident.parked:
                return 0
            del self._models[key]
            freed = resident.bytes
            residency = resident.residency
        del resident
        if residency.startswith("vram:cuda"):
            import torch

            torch.cuda.empty_cache()
        return freed

    @contextmanager
    def park(self, key: str, device: torch.device) -> Iterator[bool]:
        source_device: torch.device | None = None
        with self._available:
            resident = self._models.get(key)
            if resident is None or resident.uses or resident.parked:
                parked = None
            else:
                parked = resident
                source_device = next(resident.model.parameters()).device
                resident.parked = True
                try:
                    resident.model.to(device)
                except Exception:
                    self._models.pop(key, None)
                    resident.parked = False
                    self._available.notify_all()
                    raise
                if device.type == "cuda":
                    import torch

                    index = (
                        device.index if device.index is not None else torch.cuda.current_device()
                    )
                    resident.residency = f"vram:cuda:{index}"
                else:
                    resident.residency = "ram"
        if parked is None:
            yield False
            return
        assert source_device is not None
        if source_device.type == "cuda" and device.type != "cuda":
            import torch

            torch.cuda.empty_cache()
        try:
            yield True
        finally:
            with self._available:
                try:
                    parked.model.to(source_device)
                except Exception:
                    self._models.pop(key, None)
                    parked.parked = False
                    self._available.notify_all()
                    raise
                if source_device.type == "cuda":
                    import torch

                    index = (
                        source_device.index
                        if source_device.index is not None
                        else torch.cuda.current_device()
                    )
                    parked.residency = f"vram:cuda:{index}"
                else:
                    parked.residency = "ram"
                parked.parked = False
                self._available.notify_all()

    async def shed(self, pressure: PressureSignal) -> int:
        with self._lock:
            selected = set(pressure.items) if pressure.items is not None else None
            freed = 0
            evicted: list[_Resident] = []
            for key in list(self._models):
                if (
                    self._models[key].uses
                    or self._models[key].parked
                    or self._models[key].residency != pressure.device
                ):
                    continue
                if selected is not None and key not in selected:
                    continue
                if selected is None and freed >= pressure.bytes_needed:
                    break
                freed += self._models[key].bytes
                evicted.append(self._models.pop(key))
        del evicted
        if freed and pressure.device.startswith("vram:cuda"):
            import torch

            torch.cuda.empty_cache()
        return freed

    async def full_release(self) -> FullReleaseResult:
        """Drop every idle model while refusing active or parked entries."""
        with self._lock:
            removed = [
                self._models.pop(key)
                for key in tuple(self._models)
                if not self._models[key].uses and not self._models[key].parked
            ]
            busy = bool(self._models)
        released_cuda = any(item.residency.startswith("vram:cuda") for item in removed)
        del removed
        if released_cuda:
            import torch

            torch.cuda.empty_cache()
        return FullReleaseResult("busy" if busy else "complete")

    def details(self) -> list[ConsumerItem]:
        with self._lock:
            return [
                ConsumerItem(
                    item_id=key,
                    display_name=key,
                    bytes_by_residency={resident.residency: resident.bytes},
                )
                for key, resident in self._models.items()
            ]


MODEL_CACHE = ModelCache()


def memory_consumers() -> Mapping[str, ModelCache]:
    return {"preprocessor-models": MODEL_CACHE}


__all__ = ["MODEL_CACHE", "ModelCache", "memory_consumers"]
