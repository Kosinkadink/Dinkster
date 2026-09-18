from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer
from dinkster_server.cache_client import PeerCacheStore
from dinkster_values import TypeRegistry


def test_peer_blob_limit_rejects_manifest_size_and_lying_response() -> None:
    async def scenario() -> None:
        requests = 0

        async def blob(request: web.Request) -> web.StreamResponse:
            nonlocal requests
            requests += 1
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b"12345")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/cache/cas/{digest}", blob)
        server = TestServer(app)
        await server.start_server()
        peer = PeerCacheStore(str(server.make_url("")), TypeRegistry(), max_blob_bytes=4)
        try:
            # Oversized manifest claims are rejected before any request.
            assert await peer._fetch_blob("irrelevant", 5) is None
            assert requests == 0
            # A peer that sends more than its claimed size is read only to
            # size + 1 and rejected rather than buffered without a bound.
            assert await peer._fetch_blob("irrelevant", 4) is None
            assert requests == 1
        finally:
            await peer.close()
            await server.close()

    asyncio.run(scenario())
