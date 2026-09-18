"""PeerClient: speak the coordination surface to another instance.

The typed client half of DESIGN 3.10's cross-instance protocol. Verdicts
are relayed, never rewritten: a peer's 507 raises the same BudgetExceeded
and its 503 the same ReservationTimeout the local governor would raise, so
callers reason about one admission model whether memory is local or a
socket away. A peer that runs ungoverned answers 409, surfaced as
PeerUngoverned - a clean "no", distinct from failure.

`held()` is the cooperative-lease loop in one shape: acquire, renew at half
TTL (the liveness heartbeat), release on exit. A holder that dies simply
stops renewing and the peer's TTL reclaims the memory - the mortality the
protocol is built around.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import quote

import aiohttp
from dinkster_memory import BudgetExceeded, ReservationTimeout


class PeerError(Exception):
    """A peer answered outside the protocol (unexpected status/shape)."""


class PeerUngoverned(Exception):
    """The peer runs without a governor: coordination is not on offer."""


@dataclass(frozen=True)
class PeerLease:
    """A lease we hold on a peer, as the wire reported it."""

    lease_id: str
    device: str
    nbytes: int
    expires_in: float  # seconds remaining, as of the response


def _lease_from_wire(data: dict[str, Any]) -> PeerLease:
    lease_id = data.get("reservationId")
    device = data.get("device")
    nbytes = data.get("bytes")
    expires_in = data.get("expiresInSeconds")
    if (
        not isinstance(lease_id, str)
        or not isinstance(device, str)
        or isinstance(nbytes, bool)
        or not isinstance(nbytes, int)
        or isinstance(expires_in, bool)
        or not isinstance(expires_in, (int, float))
    ):
        raise PeerError(f"malformed lease payload: {data!r}")
    return PeerLease(
        lease_id=lease_id,
        device=device,
        nbytes=nbytes,
        expires_in=float(expires_in),
    )


class PeerClient:
    """One instance's view of one peer's coordination endpoints."""

    def __init__(
        self,
        endpoint: str,
        *,
        session: aiohttp.ClientSession | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        """``request_timeout`` bounds every call, so a dead peer answers as
        PeerError instead of a hang. ``reserve`` extends it by the server-side
        wait it explicitly asked for."""
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        self._endpoint = endpoint.rstrip("/")
        self._session = session
        self._owns_session = session is None
        self._request_timeout = request_timeout

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> PeerClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- surface --------------------------------------------------------

    async def status(self, *, details: bool = False) -> dict[str, Any]:
        """The peer's /memory/status: occupancy, governor state, leases.

        ``details`` adds per-consumer item lists (the detail contract) -
        what a memory panel renders; admission polling leaves it off.
        """
        path = "/memory/status?details=1" if details else "/memory/status"
        async with self._request("GET", path) as resp:
            return await self._json(resp, expect=200)

    async def shed(self, device: str, nbytes: int) -> int:
        """Ask the peer to free bytes; returns what it actually freed."""
        body: dict[str, object] = {"device": device, "bytes": nbytes}
        async with self._request("POST", "/memory/shed", body) as resp:
            data = await self._json(resp, expect=200)
        return self._freed(data)

    async def trim(
        self,
        device: str,
        *,
        nbytes: int | None = None,
        consumers: Sequence[str] | None = None,
        items: Sequence[str] | None = None,
    ) -> int:
        """Cache maintenance on the peer; optionally targeting consumers.

        ``items`` (stable IDs from ``status(details=True)``) narrows to
        specific things one consumer holds - "unload this model" is a trim
        naming its consumer and its item.
        """
        body: dict[str, object] = {"device": device}
        if nbytes is not None:
            body["bytes"] = nbytes
        if consumers is not None:
            body["consumers"] = list(consumers)
        if items is not None:
            body["items"] = list(items)
        async with self._request("POST", "/cache/trim", body) as resp:
            data = await self._json(resp, expect=200)
        return self._freed(data)

    async def reserve(
        self,
        device: str,
        nbytes: int,
        *,
        ttl: float | None = None,
        timeout: float | None = None,
    ) -> PeerLease:
        """Take a TTL lease on the peer; relays its governor's verdicts."""
        body: dict[str, object] = {"device": device, "bytes": nbytes}
        if ttl is not None:
            body["ttlSeconds"] = ttl
        if timeout is not None:
            body["timeoutSeconds"] = timeout
        # The peer may lawfully hold the request while it waits for memory;
        # give the HTTP call that long plus the usual protocol allowance.
        total = self._request_timeout + (timeout or 0.0)
        async with self._request("POST", "/memory/reserve", body, total=total) as resp:
            if resp.status == 507:
                raise BudgetExceeded(await self._error_text(resp))
            if resp.status == 503:
                raise ReservationTimeout(await self._error_text(resp))
            data = await self._json(resp, expect=201)
        return _lease_from_wire(data)

    async def renew(self, lease_id: str, *, ttl: float | None = None) -> PeerLease | None:
        """Heartbeat a lease; None when the peer no longer knows it."""
        body: dict[str, object] = {}
        if ttl is not None:
            body["ttlSeconds"] = ttl
        path = f"/memory/reserve/{quote(lease_id, safe='')}/renew"
        async with self._request("POST", path, body) as resp:
            if resp.status == 404:
                return None
            data = await self._json(resp, expect=200)
        return _lease_from_wire(data)

    async def release(self, lease_id: str) -> bool:
        """Release a lease; False when it already expired or was released."""
        path = f"/memory/reserve/{quote(lease_id, safe='')}"
        async with self._request("DELETE", path) as resp:
            if resp.status == 404:
                return False
            if resp.status == 409:
                raise PeerUngoverned(await self._error_text(resp))
            if resp.status != 204:
                raise PeerError(f"peer answered {resp.status} for DELETE {path}")
        return True

    @asynccontextmanager
    async def held(
        self,
        device: str,
        nbytes: int,
        *,
        ttl: float = 30.0,
        timeout: float | None = None,
    ) -> AsyncGenerator[PeerLease]:
        """Hold a lease across the body: renew at half TTL, release on exit.

        If the peer forgets the lease mid-hold (restart, expiry under
        extreme delay), renewal stops quietly - coordination is advisory,
        and the enforcement was always the peer's TTL, not ours.

        The quiet choice is marked for review (issue #70, open design
        questions): the body keeps running under a lease it may no longer
        hold. Revisit once real workloads hold leases across long operations.
        """
        lease = await self.reserve(device, nbytes, ttl=ttl, timeout=timeout)

        async def keep_renewed() -> None:
            while True:
                await asyncio.sleep(ttl / 2)
                try:
                    renewed = await self.renew(lease.lease_id, ttl=ttl)
                except (PeerError, PeerUngoverned):
                    # A flaky or restarted peer: stop renewing and let its
                    # TTL arbitrate. Advisory coordination never turns a
                    # missed heartbeat into a caller-facing crash.
                    return
                if renewed is None:
                    return

        renewer = asyncio.create_task(keep_renewed())
        try:
            yield lease
        finally:
            renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewer
            # Best-effort even when the body was cancelled: shield the
            # release so the peer frees memory now instead of at TTL expiry.
            with contextlib.suppress(PeerError, PeerUngoverned, asyncio.CancelledError):
                await asyncio.shield(self.release(lease.lease_id))

    # -- plumbing -------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        total: float | None = None,
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        return self._do_request(method, path, body, total=total)

    @asynccontextmanager
    async def _do_request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        *,
        total: float | None,
    ) -> AsyncGenerator[aiohttp.ClientResponse]:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        timeout = aiohttp.ClientTimeout(total=total or self._request_timeout)
        try:
            async with self._session.request(
                method, f"{self._endpoint}{path}", json=body, timeout=timeout
            ) as resp:
                yield resp
        except (aiohttp.ClientError, TimeoutError) as exc:
            # The wire failed, not the protocol: an unreachable, dead, or
            # silent peer surfaces as one typed verdict.
            raise PeerError(f"peer unreachable for {method} {path}: {exc!r}") from exc

    async def _json(self, resp: aiohttp.ClientResponse, *, expect: int) -> dict[str, Any]:
        if resp.status == 409:
            raise PeerUngoverned(await self._error_text(resp))
        if resp.status != expect:
            raise PeerError(
                f"peer answered {resp.status} for {resp.method} "
                f"{resp.url.path}: {await self._error_text(resp)}"
            )
        data: object = await resp.json()
        if not isinstance(data, dict):
            raise PeerError(f"peer answered a non-object body: {data!r}")
        return cast(dict[str, Any], data)

    @staticmethod
    async def _error_text(resp: aiohttp.ClientResponse) -> str:
        try:
            data: object = await resp.json()
        except Exception:
            return await resp.text()
        if isinstance(data, dict):
            error = cast("dict[str, Any]", data).get("error")
            if isinstance(error, str):
                return error
        return repr(cast(object, data))

    @staticmethod
    def _freed(data: dict[str, Any]) -> int:
        freed = data.get("freedBytes")
        if isinstance(freed, bool) or not isinstance(freed, int):
            raise PeerError(f"malformed shed/trim payload: {data!r}")
        return freed
