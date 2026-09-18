"""Best-effort projection of governor reservations into armed workers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

from dinkster_memory import MemoryGovernor

from .devices import DeviceMap

log = logging.getLogger("dinkster.workers.headroom")


class HeadroomSession(Protocol):
    async def send(
        self,
        header: Mapping[str, object],
        blobs: Sequence[bytes],
    ) -> None: ...


def _cuda_vram_index(residency: str) -> int:
    prefix = "vram:cuda:"
    index = residency.removeprefix(prefix)
    if not residency.startswith(prefix) or not index.isdigit():
        raise ValueError(f"expected vram:cuda:N residency, got {residency!r}")
    return int(index)


def worker_vram_bytes(
    values: Mapping[str, int],
    device_map: DeviceMap | None,
    *,
    diagnosed: set[tuple[str, str]] | None = None,
) -> dict[int, int]:
    """Project parent VRAM values into one worker's CUDA namespace."""
    translated: dict[int, int] = {}

    def warn_once(reason: str, residency: str, message: str) -> None:
        key = (reason, residency)
        if diagnosed is not None and key in diagnosed:
            return
        if diagnosed is not None:
            diagnosed.add(key)
        log.warning(message, residency)

    for residency, raw_nbytes in values.items():
        _cuda_vram_index(residency)
        nbytes = int(raw_nbytes)
        if nbytes < 0:
            raise ValueError("vram bytes must be non-negative")
        if device_map is not None:
            parent_device = residency.removeprefix("vram:")
            inverse_matches = [
                child for child, parent in device_map.mapping.items() if parent == parent_device
            ]
            if len(inverse_matches) > 1:
                warn_once(
                    "ambiguous",
                    residency,
                    "accelerator budget namespace translation is ambiguous for %s; "
                    "the worker receives no budget for that device",
                )
                continue
        child_residency = (
            device_map.to_child_residency(residency) if device_map is not None else residency
        )
        if child_residency is None:
            warn_once(
                "unavailable",
                residency,
                "accelerator budget namespace translation unavailable for %s; "
                "the worker receives no budget for that device",
            )
            continue
        translated[_cuda_vram_index(child_residency)] = nbytes
    return translated


def _validated_base_bytes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("base_bytes must be an integer")
    if value < 0:
        raise ValueError("base_bytes must be non-negative")
    return value


class HeadroomMirror:
    """Send live physical headroom and transient reserves to workers."""

    def __init__(self, governor: MemoryGovernor, *, base_bytes: int) -> None:
        self._base_bytes = _validated_base_bytes(base_bytes)
        self._totals: dict[str, int] = {}
        self._workers: dict[HeadroomSession, DeviceMap | None] = {}
        self._last_sent: dict[HeadroomSession, tuple[int, int]] = {}
        self._diagnosed: dict[HeadroomSession, set[tuple[str, str]]] = {}
        self._refresh_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        governor.subscribe_reserved(self._reserved_changed)

    def set_base(self, base_bytes: int) -> None:
        """Replace the process-global base and refresh every registered worker."""
        self._base_bytes = _validated_base_bytes(base_bytes)
        self._schedule_refresh()

    async def register(self, session: HeadroomSession, device_map: DeviceMap | None) -> None:
        self._workers[session] = device_map
        self._diagnosed[session] = set()
        await self._refresh()

    def deregister(self, session: HeadroomSession) -> None:
        self._workers.pop(session, None)
        self._last_sent.pop(session, None)
        self._diagnosed.pop(session, None)

    def _reserved_changed(self, device: str, total: int) -> None:
        try:
            _cuda_vram_index(device)
        except ValueError:
            return
        self._totals[device] = total
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        task = asyncio.create_task(self._refresh())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _refresh(self) -> None:
        async with self._refresh_lock:
            for session, device_map in tuple(self._workers.items()):
                translated = worker_vram_bytes(
                    self._totals,
                    device_map,
                    diagnosed=self._diagnosed[session],
                )
                extra_bytes = max(translated.values(), default=0)
                sent_value = (self._base_bytes, extra_bytes)
                if self._last_sent.get(session) == sent_value:
                    continue
                try:
                    await session.send(
                        {
                            "type": "aimdoHeadroom",
                            "extraBytes": extra_bytes,
                            "baseBytes": self._base_bytes,
                        },
                        [],
                    )
                except Exception:  # noqa: BLE001 - a dying worker is best-effort
                    continue
                if session in self._workers:
                    self._last_sent[session] = sent_value
