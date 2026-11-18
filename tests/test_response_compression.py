"""Negotiated compression preserves bytes and managed public protocols."""

from __future__ import annotations

import asyncio
import gzip
import json
import subprocess
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_server.app import create_app
from dinkster_supervisor.environment_runtime import start_engine, stop_engine
from test_environment_runtime import free_port, make_generation
from test_server import SCHEMAS, make_engine

from dinkster.frontend import install_frontend


@pytest.mark.parametrize("enabled", [False, True])
def test_http_compression_negotiates_without_changing_streams_or_media(
    tmp_path: Path, enabled: bool
) -> None:
    text = ("original frontend and API bytes " * 150).encode()
    encoded = gzip.compress(text)
    bundle = tmp_path / "frontend"
    bundle.mkdir()
    (bundle / "index.html").write_bytes(text)
    (bundle / "index.html.gz").write_bytes(gzip.compress(b"wrong compressed sibling"))

    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, response_compression=enabled)

        async def document(_request: web.Request) -> web.Response:
            return web.Response(
                body=text,
                content_type="application/json",
                headers={"Vary": "Origin", "ETag": '"original"'},
            )

        async def media(_request: web.Request) -> web.Response:
            return web.Response(body=encoded, content_type="image/png")

        async def precompressed(_request: web.Request) -> web.Response:
            return web.Response(
                body=encoded, content_type="text/plain", headers={"Content-Encoding": "gzip"}
            )

        async def no_transform(_request: web.Request) -> web.Response:
            return web.Response(
                body=text, content_type="text/plain", headers={"Cache-Control": "no-transform"}
            )

        async def stream(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(headers={"Content-Type": "text/plain"})
            await response.prepare(request)
            await response.write(text[:37])
            await response.write(text[37:])
            await response.write_eof()
            return response

        async def socket(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse(compress=False)
            await ws.prepare(request)
            message = await ws.receive_str()
            await ws.send_str("echo:" + message)
            await ws.close()
            return ws

        app.router.add_get("/api/document", document)
        app.router.add_get("/api/media", media)
        app.router.add_get("/api/precompressed", precompressed)
        app.router.add_get("/api/no-transform", no_transform)
        app.router.add_get("/api/stream", stream)
        app.router.add_get("/api/socket", socket)
        install_frontend(app, bundle=bundle)
        client = TestClient(TestServer(app), auto_decompress=False)
        await client.start_server()
        try:
            for path in ("/", "/api/document"):
                for accept, accepted in (
                    ("gzip", True),
                    ("GZip; q=0.6", True),
                    ("*;q=0.5", True),
                    ("gzip;q=0, *;q=1", False),
                    ("br", False),
                    ("identity", False),
                    ("gzip;q=0.2, identity;q=1", False),
                    ("gzip;q=invalid", False),
                    ("", False),
                ):
                    response = await client.get(path, headers={"Accept-Encoding": accept})
                    assert response.status == 200
                    body = await response.read()
                    if enabled and accepted:
                        assert response.headers["Content-Encoding"] == "gzip"
                        assert len(body) < len(text)
                        assert gzip.decompress(body) == text
                        assert response.headers["ETag"].startswith("W/")
                    else:
                        assert "Content-Encoding" not in response.headers
                        assert body == text
                    if enabled:
                        assert (
                            "accept-encoding"
                            in ",".join(response.headers.getall("Vary", [])).lower()
                        )
                    if path == "/api/document":
                        assert "Origin" in response.headers.getall("Vary", [])
            absent = await client.get("/", skip_auto_headers={"Accept-Encoding"})
            assert "Content-Encoding" not in absent.headers
            assert await absent.read() == text
            head = await client.head("/", headers={"Accept-Encoding": "gzip"})
            assert await head.read() == b""
            assert "Content-Encoding" not in head.headers
            following = await client.get("/", headers={"Accept-Encoding": "identity"})
            assert await following.read() == text
            for path, expected in (
                ("/api/media", encoded),
                ("/api/stream", text),
                ("/api/no-transform", text),
            ):
                response = await client.get(path, headers={"Accept-Encoding": "gzip"})
                assert "Content-Encoding" not in response.headers
                assert await response.read() == expected
            response = await client.get("/api/precompressed", headers={"Accept-Encoding": "gzip"})
            assert await response.read() == encoded
            assert response.headers["Content-Encoding"] == "gzip"
            response = await client.get(
                "/", headers={"Accept-Encoding": "gzip", "Range": "bytes=7-39"}
            )
            assert response.status == 206 and await response.read() == text[7:40]
            assert "Content-Encoding" not in response.headers
            async with client.ws_connect("/api/socket", headers={"Accept-Encoding": "gzip"}) as ws:
                await ws.send_str("unchanged")
                assert (await ws.receive()).data == "echo:unchanged"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_managed_compression_survives_restart_and_can_be_disabled(tmp_path: Path) -> None:
    from dinkster_registry import Lockfile

    from dinkster.generation_packs import stage_generation_packs

    generation = tmp_path / "builds/1"
    generation.mkdir(parents=True)
    stage_generation_packs(generation, Lockfile.of(()), {})
    make_generation(tmp_path, fake_serve=False)
    port = free_port()
    settings = tmp_path / "environment.json"
    settings.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": port,
            }
        )
    )
    bundle = tmp_path / "frontend"
    bundle.mkdir()
    text = ("managed response bytes " * 200).encode()
    (bundle / "index.html").write_bytes(text)
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(tmp_path),
        "compression",
    ]

    def configure(mode: str, confirmation: str = "--yes") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*command, mode, confirmation, "--json"], capture_output=True, text=True
        )

    async def probe(enabled: bool) -> None:
        async with aiohttp.ClientSession(auto_decompress=False) as client:
            for path in ("/", "/api/nodes"):
                async with client.get(
                    f"http://127.0.0.1:{port}" + path, headers={"Accept-Encoding": "identity"}
                ) as response:
                    assert response.status == 200
                    identity = await response.read()
                async with client.get(
                    f"http://127.0.0.1:{port}" + path, headers={"Accept-Encoding": "gzip"}
                ) as response:
                    assert response.status == 200
                    body = await response.read()
                    if enabled and (path == "/" or len(identity) >= 1024):
                        assert response.headers["Content-Encoding"] == "gzip"
                        assert gzip.decompress(body) == identity
                    else:
                        assert "Content-Encoding" not in response.headers
                        assert body == identity
                if path == "/":
                    assert identity == text
            async with client.get(
                f"http://127.0.0.1:{port}/", skip_auto_headers={"Accept-Encoding"}
            ) as response:
                assert "Content-Encoding" not in response.headers
                assert await response.read() == text

    before = settings.read_bytes()
    assert configure("on", "--plan").returncode == 0
    assert settings.read_bytes() == before
    try:
        assert configure("on").returncode == 0
        assert json.loads(settings.read_text())["responseCompression"] is True
        for _ in range(2):
            start_engine(tmp_path, 1, "main", port, frontend_root=str(bundle))
            asyncio.run(probe(True))
            refused = configure("off")
            assert refused.returncode != 0 and "stop all project engines" in refused.stdout
            stop_engine(tmp_path, "main")
        assert configure("off").returncode == 0
        start_engine(tmp_path, 1, "main", port, frontend_root=str(bundle))
        asyncio.run(probe(False))
    finally:
        stop_engine(tmp_path, "main")
