"""Ranked local, P2P, and HTTP asset transport."""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .identity import AssetError, require_digest
from .model import AssetResolution, AssetResolver
from .p2p_descriptor import P2PDescriptorV1, validate_p2p_descriptor
from .vault import AssetVault

TransportKind = Literal["lan-p2p", "global-p2p", "http"]
TransportHealth = Literal["healthy", "degraded", "broken"]
TransferStatus = Literal["progress", "complete", "failed"]

MAX_TRANSPORT_CANDIDATES = 128
_MAX_CANDIDATES_PER_PROVIDER = MAX_TRANSPORT_CANDIDATES // 2
DEFAULT_DISCOVERY_TIMEOUT = 3.0
DEFAULT_TRANSFER_STALL_TIMEOUT = 30.0


@dataclass(frozen=True)
class TransportCandidate:
    """One transport lead whose content still must verify by digest."""

    kind: TransportKind
    source: str
    region: str = ""
    health: TransportHealth = "healthy"
    live: bool = True
    size_bytes: int | None = None
    descriptor: P2PDescriptorV1 | None = None
    expires_at: float | None = None
    peer_address: str | None = None
    peer_port: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"lan-p2p", "global-p2p", "http"}:
            raise AssetError(f"unsupported transport kind {self.kind!r}")
        if not self.source or self.source != self.source.strip():
            raise AssetError("transport source must be a non-empty trimmed string")
        if self.health not in {"healthy", "degraded", "broken"}:
            raise AssetError(f"unsupported transport health {self.health!r}")
        if self.kind != "http" and self.region:
            raise AssetError("only HTTP transport candidates carry a region")
        if self.kind != "http" and self.health != "healthy":
            raise AssetError("P2P transport candidates do not carry HTTP health")
        if self.kind == "http" and not self.live:
            raise AssetError("HTTP transport candidates use health instead of live state")
        if self.kind == "http":
            if (
                self.size_bytes is not None
                or self.descriptor is not None
                or self.expires_at is not None
                or self.peer_address is not None
                or self.peer_port is not None
            ):
                raise AssetError("HTTP transport candidates do not carry P2P authority")
            return
        if (
            type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or not isinstance(self.descriptor, P2PDescriptorV1)
            or isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, (int, float))
            or not math.isfinite(self.expires_at)
            or self.expires_at <= 0
        ):
            raise AssetError("P2P transport candidates require bounded descriptor authority")
        if self.kind != "lan-p2p" and (self.peer_address is not None or self.peer_port is not None):
            raise AssetError("only LAN P2P candidates carry a direct peer endpoint")
        if (self.peer_address is None) != (self.peer_port is None):
            raise AssetError("LAN P2P peer address and port must be provided together")
        if self.peer_address is not None and (
            not self.peer_address
            or self.peer_address != self.peer_address.strip()
            or type(self.peer_port) is not int
            or not 1 <= self.peer_port <= 65535
        ):
            raise AssetError("LAN P2P peer endpoint is invalid")


KnownTransports = Callable[[str], Sequence[TransportCandidate]]
DiscoverLanTransports = Callable[[str], Awaitable[Sequence[TransportCandidate]]]
ValidateTransportResult = Callable[[str, TransportCandidate | None, Path], bool]


class TransportBackend(Protocol):
    """Transfer adapter owned by the active HTTP and P2P implementations."""

    def transfer(
        self,
        digest: str,
        candidate: TransportCandidate,
        *,
        stall_timeout: float,
    ) -> AsyncGenerator[TransferStatus]: ...

    async def discard_partial(self, digest: str, candidate: TransportCandidate) -> None: ...


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

    def tier(candidate: TransportCandidate) -> int | None:
        if candidate.kind == "lan-p2p":
            return 0 if candidate.live else None
        if (
            candidate.kind == "http"
            and candidate.health == "healthy"
            and preferred_region
            and candidate.region == preferred_region
        ):
            return 1
        if candidate.kind == "global-p2p":
            return 2 if candidate.live else None
        if candidate.health == "healthy":
            return 3
        if candidate.health == "degraded":
            return 4
        return 5

    ranked: list[tuple[int, int, TransportCandidate]] = []
    seen: set[tuple[str, str]] = set()
    for position, candidate in enumerate(candidates[:MAX_TRANSPORT_CANDIDATES]):
        key = (candidate.kind, candidate.source)
        candidate_tier = tier(candidate)
        if key in seen or candidate_tier is None:
            continue
        seen.add(key)
        ranked.append((candidate_tier, position, candidate))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[2] for row in ranked)


class TransportResolver:
    """Resolve locally first, then transfer through ranked bounded leads.

    The backend owns resumable staging and publication. A reported completion
    is accepted only when the canonical vault object exists and
    ``validate_complete`` confirms its descriptor, size, digest, and format.
    """

    def __init__(
        self,
        local: AssetResolver,
        vault: AssetVault,
        known_transports: KnownTransports,
        discover_lan: DiscoverLanTransports,
        backend: TransportBackend,
        validate_complete: ValidateTransportResult,
        *,
        preferred_region: str = "",
        discovery_timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
        transfer_stall_timeout: float = DEFAULT_TRANSFER_STALL_TIMEOUT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if discovery_timeout <= 0:
            raise ValueError("discovery timeout must be positive")
        if transfer_stall_timeout <= 0:
            raise ValueError("transfer stall timeout must be positive")
        self._local = local
        self._vault = vault
        self._known_transports = known_transports
        self._discover_lan = discover_lan
        self._backend = backend
        self._validate_complete = validate_complete
        self._preferred_region = preferred_region
        self._discovery_timeout = discovery_timeout
        self._transfer_stall_timeout = transfer_stall_timeout
        self._clock = clock
        self._gate_root = os.path.normcase(os.fspath(vault.root.resolve()))

    def _authorized(self, digest: str, candidate: TransportCandidate) -> bool:
        if candidate.kind == "http":
            return True
        assert candidate.descriptor is not None
        assert candidate.size_bytes is not None
        assert candidate.expires_at is not None
        if candidate.expires_at <= self._clock():
            return False
        try:
            validate_p2p_descriptor(
                candidate.descriptor,
                asset_digest=digest,
                size=candidate.size_bytes,
            )
        except AssetError:
            return False
        return True

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

    async def _discard_partial(self, digest: str, candidate: TransportCandidate) -> None:
        try:
            await asyncio.wait_for(
                self._backend.discard_partial(digest, candidate),
                self._transfer_stall_timeout,
            )
        except (AssetError, OSError, TimeoutError):
            pass

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
                known = tuple(reported_known[:_MAX_CANDIDATES_PER_PROVIDER])
            except (AssetError, OSError, TimeoutError):
                known = ()
            try:
                reported_lan = await asyncio.wait_for(
                    self._discover_lan(digest), self._discovery_timeout
                )
                discovered = tuple(reported_lan[:_MAX_CANDIDATES_PER_PROVIDER])
            except (AssetError, OSError, TimeoutError):
                discovered = ()
            candidates = rank_transport_candidates(
                (*discovered, *known), preferred_region=self._preferred_region
            )
            for candidate in candidates:
                if not self._authorized(digest, candidate):
                    continue
                try:
                    status = await self._transfer(digest, candidate)
                except (AssetError, OSError, TimeoutError):
                    status = "failed"
                if status == "stalled":
                    continue
                if status != "complete":
                    if candidate.kind != "http":
                        await self._discard_partial(digest, candidate)
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
