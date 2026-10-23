"""Ranked local and HTTP asset transport."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .identity import AssetError, require_digest
from .model import AssetResolution, AssetResolver
from .vault import AssetVault

TransportKind = Literal["http"]
TransportHealth = Literal["healthy", "degraded", "broken"]
TransferStatus = Literal["progress", "complete", "failed"]

MAX_TRANSPORT_CANDIDATES = 128
DEFAULT_TRANSFER_STALL_TIMEOUT = 30.0


@dataclass(frozen=True)
class TransportCandidate:
    """One transport lead whose content still must verify by digest."""

    kind: TransportKind
    source: str
    region: str = ""
    health: TransportHealth = "healthy"

    def __post_init__(self) -> None:
        if self.kind != "http":
            raise AssetError(f"unsupported transport kind {self.kind!r}")
        if not self.source or self.source != self.source.strip():
            raise AssetError("transport source must be a non-empty trimmed string")
        if self.health not in {"healthy", "degraded", "broken"}:
            raise AssetError(f"unsupported transport health {self.health!r}")


KnownTransports = Callable[[str], Sequence[TransportCandidate]]
ValidateTransportResult = Callable[[str, TransportCandidate | None, Path], bool]


class TransportBackend(Protocol):
    """Transfer adapter owned by the active HTTP implementation."""

    def transfer(
        self,
        digest: str,
        candidate: TransportCandidate,
        *,
        stall_timeout: float,
    ) -> AsyncGenerator[TransferStatus]: ...


@dataclass
class _DigestGate:
    lock: threading.Lock
    users: int = 0


_GATES: dict[tuple[str, str], _DigestGate] = {}
_GATES_LOCK = threading.Lock()


def rank_transport_candidates(
    candidates: Sequence[TransportCandidate],
    *,
    preferred_region: str = "",
) -> tuple[TransportCandidate, ...]:
    """Apply the transport order without changing order inside one tier."""

    def tier(candidate: TransportCandidate) -> int:
        if (
            candidate.health == "healthy"
            and preferred_region
            and candidate.region == preferred_region
        ):
            return 0
        if candidate.health == "healthy":
            return 1
        if candidate.health == "degraded":
            return 2
        return 3

    ranked: list[tuple[int, int, TransportCandidate]] = []
    seen: set[tuple[str, str]] = set()
    for position, candidate in enumerate(candidates[:MAX_TRANSPORT_CANDIDATES]):
        key = (candidate.kind, candidate.source)
        candidate_tier = tier(candidate)
        if key in seen:
            continue
        seen.add(key)
        ranked.append((candidate_tier, position, candidate))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[2] for row in ranked)


class TransportResolver:
    """Resolve locally first, then transfer through ranked bounded leads.

    The backend owns resumable staging and publication. A reported completion
    is accepted only when the canonical vault object exists and
    ``validate_complete`` confirms its size, digest, and format.
    """

    def __init__(
        self,
        local: AssetResolver,
        vault: AssetVault,
        known_transports: KnownTransports,
        backend: TransportBackend,
        validate_complete: ValidateTransportResult,
        *,
        preferred_region: str = "",
        transfer_stall_timeout: float = DEFAULT_TRANSFER_STALL_TIMEOUT,
    ) -> None:
        if transfer_stall_timeout <= 0:
            raise ValueError("transfer stall timeout must be positive")
        self._local = local
        self._vault = vault
        self._known_transports = known_transports
        self._backend = backend
        self._validate_complete = validate_complete
        self._preferred_region = preferred_region
        self._transfer_stall_timeout = transfer_stall_timeout
        self._gate_root = os.path.normcase(os.fspath(vault.root.resolve()))

    @asynccontextmanager
    async def _writer_gate(self, digest: str):
        key = (self._gate_root, digest)
        with _GATES_LOCK:
            gate = _GATES.get(key)
            if gate is None:
                gate = _DigestGate(threading.Lock())
                _GATES[key] = gate
            gate.users += 1
        acquired = False
        try:
            while not gate.lock.acquire(blocking=False):
                await asyncio.sleep(0.01)
            acquired = True
        except BaseException:
            with _GATES_LOCK:
                gate.users -= 1
                if gate.users == 0:
                    _GATES.pop(key, None)
            raise
        try:
            yield
        finally:
            if acquired:
                gate.lock.release()
            with _GATES_LOCK:
                gate.users -= 1
                if gate.users == 0:
                    _GATES.pop(key, None)

    def _local_path(self, digest: str) -> Path | None:
        paths = (self._vault.resolve(digest), self._local.resolve(digest))
        for path in paths:
            if path is None:
                continue
            try:
                if self._validate_complete(digest, None, path):
                    return path
            except (AssetError, OSError):
                continue
        return None

    async def _transfer(
        self,
        digest: str,
        candidate: TransportCandidate,
    ) -> Literal["complete", "stalled", "failed"]:
        async with aclosing(
            self._backend.transfer(
                digest,
                candidate,
                stall_timeout=self._transfer_stall_timeout,
            )
        ) as events:
            while True:
                try:
                    event = await asyncio.wait_for(anext(events), self._transfer_stall_timeout)
                except TimeoutError:
                    return "stalled"
                except StopAsyncIteration:
                    return "failed"
                if event != "progress":
                    return event

    async def resolve(self, digest: str) -> Path | None:
        require_digest(digest)
        held = self._local_path(digest)
        if held is not None:
            return held

        async with self._writer_gate(digest):
            held = self._local_path(digest)
            if held is not None:
                return held
            try:
                reported_known = self._known_transports(digest)
                known = tuple(reported_known[:MAX_TRANSPORT_CANDIDATES])
            except (AssetError, OSError, TimeoutError):
                known = ()
            candidates = rank_transport_candidates(known, preferred_region=self._preferred_region)
            for candidate in candidates:
                try:
                    status = await self._transfer(digest, candidate)
                except (AssetError, OSError, TimeoutError):
                    status = "failed"
                if status == "stalled":
                    continue
                if status != "complete":
                    continue
                path = self._vault.resolve(digest)
                if path is None:
                    continue
                try:
                    valid = self._validate_complete(digest, candidate, path)
                except (AssetError, OSError):
                    valid = False
                if valid:
                    return path
                return None
            return None

    async def resolve_asset(self, digest: str) -> AssetResolution | None:
        path = await self.resolve(digest)
        if path is None:
            return None
        resolution = self._vault.resolve_asset(digest)
        return (
            resolution
            if resolution is not None and resolution.path == path
            else AssetResolution(path)
        )
