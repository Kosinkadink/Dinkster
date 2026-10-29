from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import dinkster_workers.egress as egress_module
import pytest
from dinkster_inference import (
    GenerationRequest,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationStopConditions,
    OpenAIGenerationError,
    OpenAIGenerationProvider,
)
from dinkster_workers.egress import EgressProxy, EgressProxyError, normalize_egress_origin


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("https://API.Example.test/", "https://api.example.test"),
        ("https://api.example.test:443", "https://api.example.test"),
        ("https://api.example.test:8443", "https://api.example.test:8443"),
        ("https://[2606:4700:4700::1111]", "https://[2606:4700:4700::1111]"),
    ),
)
def test_normalize_egress_origin(value: str, expected: str) -> None:
    assert normalize_egress_origin(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        "http://api.example.test",
        "https://user:secret@api.example.test",
        "https://api.example.test/path",
        "https://api.example.test?query=yes",
        "https://api.example.test#fragment",
        "https://api.example.test:",
        "https://api.example.test:0",
        "https://[",
        "https://bad host.example",
        "https://127.0.0.1",
        "https://224.0.0.1",
        "https://[::1]",
        "https://[ff02::1]",
        "https://[64:ff9b::7f00:1]",
        "https://[64:ff9b::a00:1]",
        "https://[64:ff9b::a9fe:a9fe]",
    ),
)
def test_normalize_egress_origin_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(EgressProxyError):
        normalize_egress_origin(value)


async def _response(
    reader: asyncio.StreamReader,
) -> tuple[int, dict[str, str], bytes]:
    status_line = (await reader.readline()).decode("ascii").strip()
    status = int(status_line.split(" ", 2)[1])
    headers: dict[str, str] = {}
    while line := await reader.readline():
        if line == b"\r\n":
            break
        name, value = line.decode("ascii").split(":", 1)
        headers[name.lower()] = value.strip()
    body = await reader.readexactly(int(headers.get("content-length", "0")))
    return status, headers, body


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
def test_egress_proxy_allows_only_listed_origins_and_cleans_up(
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        if sys.platform == "win32":
            raise AssertionError("Unix-socket test ran on Windows")

        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while data := await reader.read(64 * 1024):
                    writer.write(data)
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        echo_server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = int(echo_server.sockets[0].getsockname()[1])

        async def test_resolver(host: str, resolved_port: int) -> tuple[str, ...]:
            assert host == "allowed.example.test"
            assert resolved_port == port
            return ("127.0.0.1",)

        monkeypatch.setattr(egress_module, "_resolve_public_addresses", test_resolver)
        socket_path = unix_socket_dir / "egress.sock"
        proxy = await EgressProxy.start(
            socket_path,
            (f"https://allowed.example.test:{port}",),
        )
        try:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(f"CONNECT allowed.example.test:{port} HTTP/1.1\r\n\r\n".encode("ascii"))
            await writer.drain()
            assert await reader.readuntil(b"\r\n\r\n") == (
                b"HTTP/1.1 200 Connection Established\r\n\r\n"
            )
            writer.write(b"through-the-proxy")
            await writer.drain()
            assert await reader.readexactly(len(b"through-the-proxy")) == b"through-the-proxy"
            writer.close()
            await writer.wait_closed()

            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(f"RESOLVE allowed.example.test:{port} HTTP/1.1\r\n\r\n".encode("ascii"))
            await writer.drain()
            status, _headers, body = await _response(reader)
            assert status == 200
            assert json.loads(body) == {"addresses": ["127.0.0.1"]}
            writer.close()
            await writer.wait_closed()

            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(f"CONNECT denied.example.test:{port} HTTP/1.1\r\n\r\n".encode("ascii"))
            await writer.drain()
            status, _headers, _body = await _response(reader)
            assert status == 403
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.close()
            echo_server.close()
            await echo_server.wait_closed()
        assert not socket_path.exists()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
def test_openai_transport_reaches_https_through_egress_proxy(
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        if sys.platform == "win32":
            raise AssertionError("Unix-socket test ran on Windows")
        upstream_bytes: list[bytes] = []

        async def capture(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                upstream_bytes.append(await reader.read(4096))
            finally:
                writer.close()
                await writer.wait_closed()

        upstream = await asyncio.start_server(capture, "127.0.0.1", 0)
        port = int(upstream.sockets[0].getsockname()[1])

        async def test_resolver(host: str, resolved_port: int) -> tuple[str, ...]:
            assert (host, resolved_port) == ("allowed.example.test", port)
            return ("127.0.0.1",)

        monkeypatch.setattr(egress_module, "_resolve_public_addresses", test_resolver)
        proxy = await EgressProxy.start(
            unix_socket_dir / "openai-egress.sock",
            (f"https://allowed.example.test:{port}",),
        )
        provider = OpenAIGenerationProvider(
            f"https://allowed.example.test:{port}/v1",
            "test-model",
            timeout_s=2.0,
            proxy_socket=str(proxy.socket_path),
        )
        request = GenerationRequest(
            provider.id,
            provider.model_identity,
            prompt="hello",
            sampler=GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),)),
            stop=GenerationStopConditions(1),
        )

        def execute() -> None:
            with provider.generate(request, cancelled=lambda: False) as stream:
                tuple(stream)

        try:
            with pytest.raises(OpenAIGenerationError, match="transport failed"):
                await asyncio.to_thread(execute)
            assert upstream_bytes
            assert upstream_bytes[0].startswith(b"\x16\x03")
        finally:
            provider.close()
            await proxy.close()
            upstream.close()
            await upstream.wait_closed()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
@pytest.mark.parametrize(
    "answer",
    (
        "127.0.0.1",
        "224.0.0.1",
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "64:ff9b::a9fe:a9fe",
    ),
)
def test_egress_proxy_rejects_non_public_dns_answers(
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    def private_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", (answer, 443))]

    async def scenario() -> None:
        if sys.platform == "win32":
            raise AssertionError("Unix-socket test ran on Windows")
        monkeypatch.setattr(socket, "getaddrinfo", private_getaddrinfo)
        proxy = await EgressProxy.start(
            unix_socket_dir / "egress.sock",
            ("https://private.example.test",),
        )
        try:
            reader, writer = await asyncio.open_unix_connection(proxy.socket_path)
            writer.write(b"CONNECT private.example.test:443 HTTP/1.1\r\n\r\n")
            await writer.drain()
            status, _headers, _body = await _response(reader)
            assert status == 403
            writer.close()
            await writer.wait_closed()
        finally:
            await proxy.close()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
def test_egress_proxy_bounds_idle_and_concurrent_clients(
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        if sys.platform == "win32":
            raise AssertionError("Unix-socket test ran on Windows")
        monkeypatch.setattr(egress_module, "_MAX_CLIENTS", 1)
        monkeypatch.setattr(egress_module, "_REQUEST_TIMEOUT_SECONDS", 0.05)
        proxy = await EgressProxy.start(
            unix_socket_dir / "egress.sock",
            ("https://api.example.test",),
        )
        first_reader, first_writer = await asyncio.open_unix_connection(proxy.socket_path)
        try:
            for _ in range(100):
                if proxy._clients:
                    break
                await asyncio.sleep(0.001)
            assert len(proxy._clients) == 1

            second_reader, second_writer = await asyncio.open_unix_connection(proxy.socket_path)
            try:
                assert await asyncio.wait_for(second_reader.read(), timeout=1.0) == b""
            finally:
                second_writer.close()
                await second_writer.wait_closed()

            status, _headers, _body = await asyncio.wait_for(
                _response(first_reader),
                timeout=1.0,
            )
            assert status == 502
        finally:
            first_writer.close()
            await first_writer.wait_closed()
            await proxy.close()

    asyncio.run(scenario())
