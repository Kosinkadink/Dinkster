from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from aiohttp import web
from dinkster_assets import SeedGrantV1

NetworkCost = Literal["metered", "unmetered", "unknown"]
NetworkOverride = Literal["auto", "metered", "unmetered"]
SeedAuthorizationState = Literal["active", "inactive", "revoked"]
TransferState = Literal[
    "queued", "downloading", "seeding", "paused", "stopped", "complete", "error"
]
P2PAction = Literal["pause", "resume", "stop", "remove-partial", "reset-budget", "continuous-seed"]

_DIGEST = re.compile(r"^blake3:[0-9a-f]{64}$")
_GRANT_ID = re.compile(r"^[0-9a-f]{64}$")
_ACTIONS: frozenset[str] = frozenset(
    {"pause", "resume", "stop", "remove-partial", "reset-budget", "continuous-seed"}
)


@dataclass(frozen=True, slots=True)
class P2PSeedAuthorization:
    grant_id: str
    state: SeedAuthorizationState
    grant: SeedGrantV1 | None

    def __post_init__(self) -> None:
        if type(self.grant_id) is not str or _GRANT_ID.fullmatch(self.grant_id) is None:
            raise ValueError("seed grant id must be 64 lowercase hexadecimal characters")
        if self.state not in {"active", "inactive", "revoked"}:
            raise ValueError("invalid seed authorization state")
        if self.grant is not None and self.grant.grant_id != self.grant_id:
            raise ValueError("seed authorization grant id does not match its canonical grant")
        if self.state != "revoked" and self.grant is None:
            raise ValueError("active or inactive seed authorization requires a canonical grant")
        if self.state == "revoked" and self.grant is not None:
            raise ValueError("revoked seed authorization must be a grant tombstone")

    def to_wire(self) -> dict[str, object]:
        return {
            "grantId": self.grant_id,
            "state": self.state,
            "grant": None if self.grant is None else self.grant.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class P2PTransfer:
    digest: str
    state: TransferState
    size_bytes: int
    peers: int
    download_rate_bytes_per_second: int
    upload_rate_bytes_per_second: int
    downloaded_bytes: int
    uploaded_bytes: int
    partial_bytes: int
    seed_authorizations: tuple[P2PSeedAuthorization, ...]
    remaining_seed_ratio: float | None
    remaining_seed_time_seconds: int | None

    def __post_init__(self) -> None:
        if type(self.digest) is not str or _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("invalid BLAKE3 digest")
        if self.state not in {
            "queued",
            "downloading",
            "seeding",
            "paused",
            "stopped",
            "complete",
            "error",
        }:
            raise ValueError("invalid transfer state")
        counters = (
            self.size_bytes,
            self.peers,
            self.download_rate_bytes_per_second,
            self.upload_rate_bytes_per_second,
            self.downloaded_bytes,
            self.uploaded_bytes,
            self.partial_bytes,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("P2P counters must be non-negative integers")
        object.__setattr__(self, "seed_authorizations", tuple(self.seed_authorizations))
        grant_ids = [authorization.grant_id for authorization in self.seed_authorizations]
        if len(grant_ids) != len(set(grant_ids)):
            raise ValueError("P2P seed authorization grant ids must be unique")
        if any(
            authorization.grant is not None and authorization.grant.digest != self.digest
            for authorization in self.seed_authorizations
        ):
            raise ValueError("P2P seed authorization digest must match its transfer")
        if self.remaining_seed_ratio is not None and (
            type(self.remaining_seed_ratio) is bool
            or type(self.remaining_seed_ratio) not in {int, float}
            or not math.isfinite(self.remaining_seed_ratio)
            or self.remaining_seed_ratio < 0
        ):
            raise ValueError("remaining seed ratio must be finite and non-negative")
        if self.remaining_seed_time_seconds is not None and (
            type(self.remaining_seed_time_seconds) is not int
            or self.remaining_seed_time_seconds < 0
        ):
            raise ValueError("remaining seed time must be a non-negative integer")

    def to_wire(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "state": self.state,
            "sizeBytes": self.size_bytes,
            "peers": self.peers,
            "downloadRateBytesPerSecond": self.download_rate_bytes_per_second,
            "uploadRateBytesPerSecond": self.upload_rate_bytes_per_second,
            "downloadedBytes": self.downloaded_bytes,
            "uploadedBytes": self.uploaded_bytes,
            "partialBytes": self.partial_bytes,
            "seedAuthorizations": [
                authorization.to_wire() for authorization in self.seed_authorizations
            ],
            "remainingSeedRatio": self.remaining_seed_ratio,
            "remainingSeedTimeSeconds": self.remaining_seed_time_seconds,
        }


@dataclass(frozen=True, slots=True)
class P2PNetworkStatus:
    system: NetworkCost
    override: NetworkOverride
    effective: NetworkCost
    paused: bool

    def __post_init__(self) -> None:
        if self.system not in {"metered", "unmetered", "unknown"}:
            raise ValueError("invalid system network cost")
        if self.override not in {"auto", "metered", "unmetered"}:
            raise ValueError("invalid network override")
        if self.effective not in {"metered", "unmetered", "unknown"}:
            raise ValueError("invalid effective network cost")
        if type(self.paused) is not bool:
            raise ValueError("network pause state must be a boolean")
        expected = self.system if self.override == "auto" else self.override
        if self.effective != expected:
            raise ValueError("effective network cost must match the system cost or override")

    def to_wire(self) -> dict[str, object]:
        return {
            "system": self.system,
            "override": self.override,
            "effective": self.effective,
            "paused": self.paused,
        }


@dataclass(frozen=True, slots=True)
class P2PTotals:
    downloaded_bytes: int
    uploaded_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.downloaded_bytes, self.uploaded_bytes)
        ):
            raise ValueError("P2P totals must be non-negative integers")

    def to_wire(self) -> dict[str, int]:
        return {
            "downloadedBytes": self.downloaded_bytes,
            "uploadedBytes": self.uploaded_bytes,
        }


@dataclass(frozen=True, slots=True)
class P2PActivityStatus:
    network: P2PNetworkStatus
    totals: P2PTotals
    transfers: tuple[P2PTransfer, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transfers", tuple(self.transfers))
        digests = [transfer.digest for transfer in self.transfers]
        if len(digests) != len(set(digests)):
            raise ValueError("P2P transfer digests must be unique")

    def to_wire(self) -> dict[str, object]:
        return {
            "network": self.network.to_wire(),
            "totals": self.totals.to_wire(),
            "transfers": [transfer.to_wire() for transfer in self.transfers],
        }


class P2PActivityProvider(Protocol):
    async def perform_action(self, digest: str, action: P2PAction) -> None: ...


class P2PProviderError(Exception):
    pass


class P2PProviderConflict(P2PProviderError):
    pass


class P2PProviderUnavailable(P2PProviderError):
    pass


class P2PTransferNotFound(P2PProviderError):
    pass


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


def add_p2p_action_routes(
    app: web.Application,
    provider: P2PActivityProvider | None,
    *,
    p2p_granted: bool,
) -> None:
    async def perform(request: web.Request) -> web.Response:
        digest, action = request.match_info["digest"], request.match_info["action"]
        if not p2p_granted:
            return _error(403, "p2p-permission-denied", "P2P actions are disabled by this host")
        if _DIGEST.fullmatch(digest) is None:
            return _error(400, "invalid-p2p-digest", "digest must be a canonical BLAKE3 digest")
        if action not in _ACTIONS:
            return _error(400, "invalid-p2p-action", "unsupported P2P transfer action")
        if provider is None:
            return _error(503, "p2p-unavailable", "P2P activity is unavailable")
        try:
            await provider.perform_action(digest, cast(P2PAction, action))
            return web.Response(status=204)
        except P2PProviderConflict:
            return _error(409, "p2p-conflict", "P2P action conflicts with current state")
        except P2PProviderUnavailable:
            return _error(503, "p2p-unavailable", "P2P activity is unavailable")
        except P2PTransferNotFound:
            return _error(404, "p2p-transfer-not-found", "P2P transfer was not found")
        except Exception:
            return _error(500, "p2p-internal-error", "P2P activity failed")

    app.router.add_post("/api/p2p/transfers/{digest}/{action}", perform)
