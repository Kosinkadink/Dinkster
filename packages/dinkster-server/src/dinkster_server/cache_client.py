"""PeerCacheStore: another instance's cache as a read-only layer.

A CacheStore (structurally) whose get() pulls from a peer's cache-sharing
endpoints: fetch the entry manifest, then each payload blob, verify every
blob against its digest, and rehydrate through the same wire code the disk
store uses - a peer hit and a disk hit are literally one function apart.

Read-only by design: put() is a no-op. Nothing writes into another
instance's cache; an instance's persistent cache is populated only by its
own engine. Compose this store behind local layers (LayeredCache promotes
a peer hit onto local memory/disk), so each entry crosses the network at
most once.

Conservative on every failure: network errors, timeouts, malformed
manifests, digest mismatches - all of it is a miss, never an exception on
the engine's execute path and never a corrupt Value. Fingerprints travel
in the manifest verbatim, so a peer hit keys identically here (hazard H4).

Trust: same domain as workers (the manifest and blobs carry codec bytes,
possibly pickle). Point this only at peers you would hand a worker token
to, over a network you trust or a tunnel you authenticated.
"""

from __future__ import annotations
from dinkster_values import MEBIBYTE

import json
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import quote

import aiohttp
from dinkster_assets import digest_bytes
from dinkster_caches import entry_from_wire
from dinkster_protocol import CacheKey
from dinkster_values import PEER_CACHE_BLOB_LIMIT_BYTES, TypeRegistry, Value

_MAX_MANIFEST_BYTES = 4 * MEBIBYTE
DEFAULT_MAX_BLOB_BYTES = PEER_CACHE_BLOB_LIMIT_BYTES


class PeerCacheStore:
    """One instance's read-only view of one peer's shared cache."""

    def __init__(
        self,
        endpoint: str,
        registry: TypeRegistry,
        *,
        session: aiohttp.ClientSession | None = None,
        request_timeout: float = 30.0,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        if max_blob_bytes < 1:
            raise ValueError("max_blob_bytes must be >= 1")
        self._endpoint = endpoint.rstrip("/")
        self._registry = registry
        self._session = session
        self._owns_session = session is None
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._max_blob_bytes = max_blob_bytes
        self.hits = 0
        self.misses = 0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> PeerCacheStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    # -- CacheStore -------------------------------------------------------

    async def get(self, key: CacheKey) -> Mapping[str, Value] | None:
        try:
            async with self._http().get(
                f"{self._endpoint}/cache/entry/{quote(key, safe='')}",
                timeout=self._timeout,
            ) as resp:
                if resp.status != 200:
                    self.misses += 1
                    return None
                if resp.content_length is not None and resp.content_length > _MAX_MANIFEST_BYTES:
                    self.misses += 1
                    return None
                # Enforce the cap on actual bytes too: a chunked response
                # declares no content length.
                raw = await resp.content.read(_MAX_MANIFEST_BYTES + 1)
                if len(raw) > _MAX_MANIFEST_BYTES:
                    self.misses += 1
                    return None
                manifest = cast(object, json.loads(raw))
            if not isinstance(manifest, Mapping):
                self.misses += 1
                return None
            entry = await entry_from_wire(
                cast("Mapping[str, Any]", manifest), self._fetch_blob, self._registry
            )
        except (aiohttp.ClientError, TimeoutError, ValueError):
            self.misses += 1
            return None
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    async def put(self, key: CacheKey, outputs: Mapping[str, Value]) -> None:
        """Read-only: peers pull from each other; nobody pushes."""

    # -- internals --------------------------------------------------------

    async def _fetch_blob(self, digest: str, size: int) -> bytes | None:
        if size < 0 or size > self._max_blob_bytes:
            return None
        try:
            async with self._http().get(
                f"{self._endpoint}/cache/cas/{quote(digest, safe='')}",
                timeout=self._timeout,
            ) as resp:
                if resp.status != 200:
                    return None
                if resp.content_length is not None and resp.content_length != size:
                    return None
                data = await resp.content.read(size + 1)
        except (aiohttp.ClientError, TimeoutError):
            return None
        # Verify against the digest the manifest named: bytes from the wire
        # are only trusted to be what they claim to be, never assumed.
        if len(data) != size or digest_bytes(data) != digest:
            return None
        return data
