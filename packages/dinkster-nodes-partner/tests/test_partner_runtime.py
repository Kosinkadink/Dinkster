from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import ssl
import sys
import tomllib
from collections.abc import AsyncIterator, Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import dinkster_workers.egress as egress_module
import numpy as np
import pytest
from dinkster_workers.egress import EgressProxy
from PIL import Image

PACK_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACK_ROOT / "src"))

from dinkster_nodes_partner.opspec import (  # noqa: E402
    ADAPTER_KINDS,
    Adapter,
    BatchMapJoin,
    Check,
    CheckInputs,
    Cond,
    DownloadDecode,
    EncodeMedia,
    FixedField,
    FormatField,
    HttpSyncBinary,
    HttpSyncJson,
    InputBinding,
    LocalProgress,
    MaskPrepare,
    MediaConstraints,
    MultipartMap,
    MultiStage,
    OpSpec,
    ProxyUpload,
    ResponseSelect,
    Segment,
    SubmitPoll,
    ValueConstruct,
)
from dinkster_nodes_partner.partner_runtime import (  # noqa: E402
    EGRESS_PROXY_ENV,
    ApiServerError,
    HttpxTransport,
    LocalNetworkError,
    MissingApiKeyError,
    OperationCancelled,
    PartnerError,
    RuntimeContext,
    TransportNetworkError,
    TrustPolicyError,
    _retry_after,
    download_bytes,
    run_op,
)


class FixtureResponse:
    def __init__(
        self,
        status: int = 200,
        *,
        headers: Mapping[str, str] | None = None,
        payload: object = None,
        chunks: tuple[bytes, ...] = (),
    ) -> None:
        self.status = status
        self.headers = dict(headers or {})
        self.payload = {} if payload is None else payload
        self.chunks = chunks
        self.closed = False

    async def json(self) -> object:
        await asyncio.sleep(0)
        return self.payload

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size == 1024 * 1024
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk

    async def close(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


class FixtureTransport:
    def __init__(self, *results: FixtureResponse | Exception) -> None:
        self.results = list(results)
        self.requests: list[tuple[str, str, dict[str, str], object]] = []
        self.byte_requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self.resolutions: dict[str, tuple[str, ...]] = {}
        self.internet = True
        self.on_request = None
        self.on_byte_request = None

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        await asyncio.sleep(0)
        return self.resolutions.get(host, ("203.0.113.10",))

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json_body: Mapping[str, object] | None,
        timeout: float,
    ) -> FixtureResponse:
        await asyncio.sleep(0)
        self.requests.append((method, url, dict(headers), json_body))
        if self.on_request is not None:
            self.on_request()
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def request_bytes(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> FixtureResponse:
        await asyncio.sleep(0)
        self.byte_requests.append((method, url, dict(headers), body))
        if self.on_byte_request is not None:
            self.on_byte_request()
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def internet_accessible(self) -> bool:
        await asyncio.sleep(0)
        return self.internet

    async def close(self) -> None:
        await asyncio.sleep(0)


def execute(spec: OpSpec, ctx: RuntimeContext, inputs: Mapping[str, object] | None = None):
    return asyncio.run(run_op(spec, inputs or {}, ctx))


def request_spec(**changes: object) -> OpSpec:
    values: dict[str, object] = {
        "id": "request",
        "path": "/proxy/example/run",
        "body": (InputBinding("prompt", "text"),),
        "max_retries": 0,
    }
    values.update(changes)
    return OpSpec(
        (
            HttpSyncJson(**values),
            ResponseSelect("select", "request", ("result", "id"), "task_id"),
        )
    )


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
def test_httpx_transport_uses_the_worker_egress_socket(
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        requests: list[bytes] = []

        async def proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            request = await reader.readuntil(b"\r\n\r\n")
            requests.append(request)
            if request.startswith(b"RESOLVE "):
                body = b'{"addresses":["1.1.1.1"]}'
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(body)).encode("ascii")
                    + b"\r\nConnection: close\r\n\r\n"
                    + body
                )
            else:
                writer.write(
                    b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        socket_path = unix_socket_dir / "egress.sock"
        server = await asyncio.start_unix_server(proxy, path=socket_path)
        monkeypatch.setenv(EGRESS_PROXY_ENV, str(socket_path))
        transport = HttpxTransport()
        try:
            assert await transport.resolve("api.example.test", 443) == ("1.1.1.1",)
            with pytest.raises(TransportNetworkError):
                await transport.request(
                    "GET",
                    "https://api.example.test/v1/status",
                    headers={"Accept": "application/json"},
                    json_body=None,
                    timeout=5.0,
                )
        finally:
            await transport.close()
            server.close()
            await server.wait_closed()
        assert requests[0].startswith(b"RESOLVE api.example.test:443 HTTP/1.1\r\n")
        assert requests[1].startswith(b"CONNECT api.example.test:443 HTTP/1.1\r\n")

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="requires Unix sockets")
def test_httpx_transport_completes_https_through_the_egress_proxy(
    unix_socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        certificate = PACK_ROOT.parents[1] / "tests" / "tls" / "service-cert.pem"
        key = PACK_ROOT.parents[1] / "tests" / "tls" / "service-key.pem"
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, key)
        requests: list[tuple[bytes, bytes]] = []

        async def https_server(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            head = await reader.readuntil(b"\r\n\r\n")
            headers = {
                name.lower(): value.strip()
                for line in head.decode("ascii").split("\r\n")[1:]
                if line
                for name, value in (line.split(":", 1),)
            }
            request_body = await reader.readexactly(int(headers.get("content-length", "0")))
            requests.append((head, request_body))
            body = b'{"proxied":true}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(https_server, "127.0.0.1", 0, ssl=tls)
        port = int(server.sockets[0].getsockname()[1])

        async def test_resolver(host: str, resolved_port: int) -> tuple[str, ...]:
            assert host == "localhost"
            assert resolved_port == port
            return ("127.0.0.1",)

        monkeypatch.setattr(egress_module, "_resolve_public_addresses", test_resolver)
        proxy = await EgressProxy.start(
            unix_socket_dir / "egress.sock",
            (f"https://localhost:{port}",),
        )
        monkeypatch.setenv(EGRESS_PROXY_ENV, str(proxy.socket_path))
        monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
        transport = HttpxTransport()
        response = None
        try:
            response = await transport.request(
                "POST",
                f"https://localhost:{port}/status",
                headers={"Accept": "application/json"},
                json_body={"hello": "proxy"},
                timeout=5.0,
            )
            assert response.status == 200
            assert await response.json() == {"proxied": True}
            await response.close()
            response = await transport.request_bytes(
                "PUT",
                f"https://localhost:{port}/upload",
                headers={"Content-Type": "application/octet-stream"},
                body=b"upload-through-proxy",
                timeout=5.0,
            )
            assert response.status == 200
            assert await response.json() == {"proxied": True}
        finally:
            if response is not None:
                await response.close()
            await transport.close()
            await proxy.close()
            server.close()
            await server.wait_closed()
        assert requests[0][0].startswith(b"POST /status HTTP/1.1\r\n")
        assert requests[0][1] == b'{"hello":"proxy"}'
        assert requests[1][0].startswith(b"PUT /upload HTTP/1.1\r\n")
        assert requests[1][1] == b"upload-through-proxy"

    asyncio.run(scenario())


def test_opspec_full_vocabulary_validates_freezes_and_round_trips_losslessly() -> None:
    adapters = (
        CheckInputs("check", (Check("required", (), (Cond("source", "present"),)),)),
        ValueConstruct("value", "value"),
        HttpSyncJson("json", "/proxy/x", body=(InputBinding("x", "source"),)),
        HttpSyncBinary("binary"),
        SubmitPoll("poll"),
        MultiStage("stages"),
        ProxyUpload("upload"),
        EncodeMedia("encode"),
        MediaConstraints("constraints"),
        MaskPrepare("mask"),
        MultipartMap("multipart"),
        DownloadDecode("download"),
        BatchMapJoin("batch", "source", "value"),
        ResponseSelect("select", "json", ("items", 0), "result"),
        LocalProgress("progress", "Done", 1, 1),
    )
    spec = OpSpec(adapters)
    assert tuple(adapter.kind for adapter in adapters) == ADAPTER_KINDS
    assert OpSpec.from_json(spec.to_json()) == spec
    with pytest.raises(FrozenInstanceError):
        spec.version = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown earlier adapter"):
        OpSpec((ResponseSelect("bad", "missing", ("x",), "out"),))


def test_amendments_b_d_new_fields_default_when_loading_prior_version_one_json() -> None:
    spec = OpSpec(
        (
            HttpSyncJson("request", "/proxy/x", body=(InputBinding("x", "x"),)),
            SubmitPoll("poll", "request"),
        )
    )
    data = json.loads(spec.to_json())
    binding = data["adapters"][0]["body"][0]
    del binding["string_case"]
    del binding["omit_if"]
    poll = data["adapters"][1]
    del poll["path_template"]
    del poll["path_value_path"]
    assert OpSpec.from_json(json.dumps(data)) == spec


def test_shared_grammar_value_map_freezes_round_trips_and_applies_precedence() -> None:
    mapping = {"ON": "LOWER", "true": 1, "false": 0}
    binding = InputBinding("result", "choice", string_case="lower", value_map=mapping)
    mapping["ON"] = "changed"
    spec = OpSpec((ValueConstruct("value", "value", (binding,)),))
    assert execute(spec, RuntimeContext(transport=FixtureTransport()), {"choice": "ON"}) == {
        "value": {"result": "lower"}
    }
    assert OpSpec.from_json(spec.to_json()) == spec
    assert binding.value_map is not None
    with pytest.raises(TypeError):
        binding.value_map["ON"] = "changed"  # type: ignore[index]
    bool_mapping = {"true": 1, "false": 0}
    assert set(bool_mapping) == {"true", "false"}
    bool_spec = OpSpec(
        (
            ValueConstruct(
                "value", "value", (InputBinding("result", "choice", value_map=bool_mapping),)
            ),
        )
    )
    assert execute(bool_spec, RuntimeContext(transport=FixtureTransport()), {"choice": True}) == {
        "value": {"result": 1}
    }
    assert execute(bool_spec, RuntimeContext(transport=FixtureTransport()), {"choice": False}) == {
        "value": {"result": 0}
    }
    with pytest.raises(PartnerError, match="result.*requires a string or boolean"):
        execute(bool_spec, RuntimeContext(transport=FixtureTransport()), {"choice": 1})
    with pytest.raises(PartnerError, match="result.*missing.*choice"):
        execute(spec, RuntimeContext(transport=FixtureTransport()), {"choice": "missing"})
    with pytest.raises(ValueError, match="must not be empty"):
        InputBinding("x", "x", value_map={})
    with pytest.raises(ValueError, match="expanded bindings"):
        InputBinding("", "x", expand=True, value_map={"x": "y"})


def test_shared_grammar_candidate_paths_skip_missing_null_and_empty_list() -> None:
    spec = OpSpec(
        (
            ValueConstruct(
                "source",
                "unused",
                bindings=(InputBinding("a", "a"), InputBinding("b", "b"), InputBinding("c", "c")),
            ),
            ResponseSelect(
                "select", "source", (), "result", paths=(("missing",), ("a",), ("b",), ("c",))
            ),
        )
    )
    assert (
        execute(
            spec,
            RuntimeContext(transport=FixtureTransport()),
            {"a": None, "b": [], "c": "hit"},
        )["result"]
        == "hit"
    )
    assert OpSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError, match="unique"):
        ResponseSelect("x", "source", (), "out", paths=(("a",), ("a",)))


def test_shared_grammar_candidate_multi_download_uses_first_nonempty_list() -> None:
    stream = io.BytesIO()
    Image.new("RGB", (2, 1), (9, 8, 7)).save(stream, "PNG")
    transport = FixtureTransport(
        FixtureResponse(headers={"Content-Type": "image/png"}, chunks=(stream.getvalue(),))
    )
    spec = OpSpec(
        (
            ValueConstruct(
                "response",
                "raw",
                (InputBinding("series_images", "series"), InputBinding("images", "images")),
            ),
            DownloadDecode(
                "download",
                "response",
                output="image",
                media_family="image",
                item_url_path=("url",),
                url_paths=(("series_images",), ("images",)),
            ),
        )
    )
    result = execute(
        spec,
        RuntimeContext(transport=transport),
        {"series": [], "images": [{"url": "https://cdn.test/image.png"}]},
    )
    assert result["image"].shape == (1, 1, 2, 3)
    assert OpSpec.from_json(spec.to_json()) == spec


def test_shared_grammar_single_candidate_path_matches_legacy_selection() -> None:
    source = ValueConstruct("source", "raw", (InputBinding("value", "value"),))
    legacy = OpSpec((source, ResponseSelect("select", "source", ("value",), "result")))
    candidate = OpSpec(
        (source, ResponseSelect("select", "source", (), "result", paths=(("value",),)))
    )
    context = RuntimeContext(transport=FixtureTransport())
    inputs = {"value": "same"}
    assert execute(legacy, context, inputs) == execute(candidate, context, inputs)
    with pytest.raises(PartnerError, match="unavailable.*missing"):
        execute(
            OpSpec(
                (
                    ValueConstruct("source", "raw"),
                    ResponseSelect(
                        "select", "source", (), "result", paths=(("missing",), ("also",))
                    ),
                )
            ),
            context,
        )


def test_shared_grammar_segment_modes_order_bounds_and_round_trip() -> None:
    join = BatchMapJoin(
        "join",
        segments=(
            Segment("first", {"kind": "image"}),
            Segment("optional", mode="single_optional"),
            Segment("many", {"kind": "video"}, mode="mapping"),
            Segment("raw", wrap_key=None, mode="verbatim"),
        ),
        min_items=4,
        max_items=4,
        min_message="need four, got {count}",
        max_message="too many: {count}",
    )
    spec = OpSpec((join, ValueConstruct("out", "out", (InputBinding("items", "join"),))))
    result = execute(
        spec,
        RuntimeContext(transport=FixtureTransport()),
        {"first": "a", "optional": None, "many": {"x": "b", "y": "c"}, "raw": {"raw": 4}},
    )
    assert result["out"]["items"] == [
        {"kind": "image", "url": "a"},
        {"kind": "video", "url": "b"},
        {"kind": "video", "url": "c"},
        {"raw": 4},
    ]
    assert OpSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError, match="verbatim"):
        Segment("raw", mode="verbatim")
    with pytest.raises(PartnerError, match="need four, got 3"):
        execute(
            spec,
            RuntimeContext(transport=FixtureTransport()),
            {"first": "a", "many": {"x": "b"}, "raw": {"r": 1}},
        )


def test_shared_grammar_segment_frozen_validation_and_runtime_refusals() -> None:
    fixed = {"kind": "image"}
    segment = Segment("item", fixed)
    fixed["kind"] = "changed"
    assert segment.fixed == {"kind": "image"}
    with pytest.raises(TypeError):
        segment.fixed["kind"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="wrap_key"):
        Segment("item", {"url": "collision"})
    with pytest.raises(ValueError, match="strings"):
        Segment("item", {"kind": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="verbatim"):
        Segment("raw", {"kind": "raw"}, None, "verbatim")
    with pytest.raises(ValueError, match="verbatim"):
        Segment("raw", wrap_key="url", mode="verbatim_optional")
    with pytest.raises(ValueError, match="exactly one"):
        BatchMapJoin("join")
    with pytest.raises(ValueError, match="exactly one"):
        BatchMapJoin("join", "source", "url", segments=(segment,))
    with pytest.raises(ValueError, match="non-negative"):
        BatchMapJoin("join", segments=(segment,), min_items=-1)
    with pytest.raises(ValueError, match="segments mode"):
        BatchMapJoin("join", "source", "url", min_message="unused")
    with pytest.raises(ValueError, match="invalid"):
        BatchMapJoin("join", segments=(segment,), min_items=2, max_items=1)
    with pytest.raises(ValueError, match="placeholder"):
        BatchMapJoin("join", segments=(segment,), min_message="bad {other}")
    with pytest.raises(PartnerError, match="must not be null"):
        execute(
            OpSpec((BatchMapJoin("join", segments=(Segment("item"),)),)),
            RuntimeContext(transport=FixtureTransport()),
            {"item": None},
        )
    with pytest.raises(PartnerError, match="key 'bad'.*string"):
        execute(
            OpSpec((BatchMapJoin("join", segments=(Segment("many", mode="mapping"),)),)),
            RuntimeContext(transport=FixtureTransport()),
            {"many": {"bad": 1}},
        )
    with pytest.raises(PartnerError, match="raw.*mapping"):
        execute(
            OpSpec(
                (BatchMapJoin("join", segments=(Segment("raw", wrap_key=None, mode="verbatim"),)),)
            ),
            RuntimeContext(transport=FixtureTransport()),
            {"raw": "not-a-mapping"},
        )
    assert (
        execute(
            OpSpec(
                (
                    BatchMapJoin(
                        "join",
                        segments=(Segment("raw", wrap_key=None, mode="verbatim_optional"),),
                    ),
                )
            ),
            RuntimeContext(transport=FixtureTransport()),
            {"raw": None},
        )
        == {}
    )


def test_amendment_m_check_count_message_frozen_runtime_fallback_and_round_trip() -> None:
    check = Check(
        "Expected 1 or 2 input images, but got {count}.",
        require=(Cond("image", "count_ge", 1), Cond("image", "count_le", 2)),
    )
    spec = OpSpec((CheckInputs("count", (check,)),))
    assert OpSpec.from_json(spec.to_json()) == spec
    with pytest.raises(FrozenInstanceError):
        check.message = "changed"  # type: ignore[misc]

    context = RuntimeContext(transport=FixtureTransport())
    for inputs, count in (({}, 0), ({"image": np.zeros((3, 1, 1, 3))}, 3)):
        with pytest.raises(PartnerError, match=rf"Expected 1 or 2 input images, but got {count}\."):
            execute(spec, context, inputs)
    with pytest.raises(PartnerError, match="got 3"):
        execute(
            spec,
            context,
            {"image.first": "a", "image.second": "b", "image.third": "c"},
        )
    assert execute(spec, context, {"image": np.zeros((2, 1, 1, 3))}) == {}


def test_amendment_m_check_count_message_closed_grammar_and_mixed_require_semantics() -> None:
    count = Cond("images", "count_le", 2)
    with pytest.raises(ValueError, match="only one optional"):
        Check("{count} {count}", require=(count,))
    with pytest.raises(ValueError, match="only one optional"):
        Check("bad {other}", require=(count,))
    with pytest.raises(ValueError, match="exactly one input"):
        Check("got {count}", require=(Cond("ready", "present"),))
    with pytest.raises(ValueError, match="exactly one input"):
        Check(
            "got {count}",
            require=(Cond("images", "count_le", 2), Cond("other", "count_ge", 1)),
        )
    check = Check(
        "images={count}",
        require=(Cond("images", "count_le", 2), Cond("ready", "present")),
    )
    with pytest.raises(PartnerError, match="images=2"):
        execute(
            OpSpec((CheckInputs("check", (check,)),)),
            RuntimeContext(transport=FixtureTransport()),
            {"images": np.zeros((2, 1, 1, 3))},
        )


def test_authorized_runtime_f_through_l_round_trip_and_core_semantics() -> None:
    formatted = FormatField("size", "{width}*{height}")
    spec = OpSpec(
        (
            ValueConstruct("body", "body", formatted=(formatted,)),
            BatchMapJoin(
                "values",
                segments=(Segment("encoded", wrap_key=None, mode="mapping_values"),),
            ),
        )
    )
    assert OpSpec.from_json(spec.to_json()) == spec
    result = execute(
        spec,
        RuntimeContext(transport=FixtureTransport()),
        {"width": 640, "height": "480", "encoded": {"a": "one", "b": None, "c": "two"}},
    )
    assert result == {"body": {"size": "640*480"}}
    for bad in (True, 1.5, None):
        with pytest.raises(PartnerError, match="width.*size|size.*width"):
            execute(spec, RuntimeContext(transport=FixtureTransport()), {"width": bad, "height": 1})
    with pytest.raises(ValueError, match="bare"):
        FormatField("x", "{x!r}")
    with pytest.raises(ValueError, match="mapping_values"):
        Segment("x", mode="mapping_values")

    optional_upload = OpSpec((ProxyUpload("upload", "missing", optional=True),))
    transport = FixtureTransport()
    execute(optional_upload, RuntimeContext(transport=transport))
    assert transport.requests == [] and transport.byte_requests == []
    assert OpSpec.from_json(optional_upload.to_json()) == optional_upload


def test_authorized_runtime_media_constraints_optional_batch_audio_and_aspect() -> None:
    context = RuntimeContext(transport=FixtureTransport())
    optional = OpSpec((MediaConstraints("check", "audio", optional=True),))
    assert execute(optional, context) == {}
    audio = MediaConstraints(
        "audio", "audio", min_duration=1.5, max_duration=2.0, duration_media="audio"
    )
    assert OpSpec.from_json(OpSpec((audio,)).to_json()) == OpSpec((audio,))
    execute(
        OpSpec((audio,)),
        context,
        {"audio": {"waveform": np.zeros((1, 1, 24000), dtype=np.float32), "sample_rate": 12000}},
    )
    with pytest.raises(PartnerError, match=r"at least 1.5s, got 1.00s"):
        execute(
            OpSpec((audio,)),
            context,
            {"audio": {"waveform": np.zeros((1, 1, 11998)), "sample_rate": 12000}},
        )
    batch = MediaConstraints(
        "images", "images", min_width=4, min_aspect_ratio=0.5, max_aspect_ratio=2, batch=True
    )
    execute(OpSpec((batch,)), context, {"images": {"first": np.zeros((4, 8, 3))}})
    with pytest.raises(PartnerError, match="second.*aspect ratio"):
        execute(
            OpSpec((batch,)),
            context,
            {"images": {"first": np.zeros((4, 8, 3)), "second": np.zeros((4, 9, 3))}},
        )
    with pytest.raises(ValueError, match="batch audio"):
        MediaConstraints("bad", "x", min_duration=1, duration_media="audio", batch=True)


def test_authorized_runtime_audio_mp3_data_url_table() -> None:
    step = EncodeMedia("audio", "audio", media_family="audio", format="MP3", output="data_url")
    assert OpSpec.from_json(OpSpec((step,)).to_json()) == OpSpec((step,))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda data: data["adapters"][0]["body"][0].update(expand="false"),
        lambda data: data["adapters"][0].update(paths=[["only-one"]]),
        lambda data: data["adapters"][0].update(fixed=["not-an-object"]),
        lambda data: data["adapters"][1].update(optional="false"),
        lambda data: data["adapters"][1].update(max_pixels="large"),
        lambda data: data["adapters"][2].update(url_path=[True]),
        lambda data: data["adapters"][3].update(min_aspect_ratio="small"),
        lambda data: data["adapters"][4].update(url_path=[{}]),
    ),
)
def test_opspec_json_import_rejects_malformed_new_descriptor_fields(mutation) -> None:
    data = json.loads(
        OpSpec(
            (
                HttpSyncJson("request", "/proxy/x", body=(InputBinding("x", "x"),)),
                EncodeMedia("encode", "image"),
                SubmitPoll("poll", "request"),
                MediaConstraints("constraint", "image"),
                DownloadDecode("download", "request", ("url",), "image"),
            )
        ).to_json()
    )
    mutation(data)
    with pytest.raises(ValueError):
        OpSpec.from_json(json.dumps(data))


def test_http_sync_json_relative_mapping_headers_response_select_and_progress() -> None:
    response = FixtureResponse(
        headers={"X-Comfy-Credits-Used": "2.5"},
        payload={"result": {"id": "task-1"}},
    )
    transport = FixtureTransport(response)
    progress: list[tuple[int, int, str]] = []
    spec = OpSpec(
        (
            LocalProgress("start", "Submitting", 0, 1),
            HttpSyncJson(
                "request",
                "/proxy/example/run",
                body=(InputBinding("prompt", "text"), InputBinding("seed", "seed")),
            ),
            ResponseSelect("select", "request", ("result", "id"), "task_id"),
        )
    )
    result = execute(
        spec,
        RuntimeContext(
            transport=transport,
            api_key="secret",
            api_base="https://proxy.test/root",
            job_id="job-7",
            progress=lambda step, total, text: progress.append((step, total, text)),
        ),
        {"text": "hello", "seed": 4},
    )
    assert result == {"task_id": "task-1"}
    method, url, headers, body = transport.requests[0]
    assert (method, url, body) == (
        "POST",
        "https://proxy.test/root/proxy/example/run",
        {"prompt": "hello", "seed": 4},
    )
    assert headers["X-API-KEY"] == "secret"
    assert headers["Comfy-Job-Id"] == "job-7"
    assert progress == [(0, 1, "Submitting"), (1, 1, "Credits used: 2.5")]
    assert response.closed


def test_absolute_request_strips_comfy_headers_and_validates_dns() -> None:
    transport = FixtureTransport(FixtureResponse(payload={"result": {"id": "ok"}}))
    spec = request_spec(path="https://provider.test/task")
    execute(spec, RuntimeContext(transport=transport, api_key="secret"), {"text": "x"})
    _, _, headers, _ = transport.requests[0]
    assert headers == {"Accept": "application/json"}


@pytest.mark.parametrize("path", ("https://private.test/task", "//private.test/task"))
def test_variant_absolute_request_uses_trust_policy(path: str) -> None:
    transport = FixtureTransport()
    transport.resolutions["private.test"] = ("127.0.0.1",)
    spec = request_spec(path_input="variant", paths=(("bad", path),))
    with pytest.raises(TrustPolicyError, match="private.test"):
        execute(
            spec,
            RuntimeContext(transport=transport, api_key="secret"),
            {"text": "x", "variant": "bad"},
        )
    assert transport.requests == []


def test_keyless_pack_loads_but_relative_execution_fails_friendly() -> None:
    with open(PACK_ROOT / "dinkster-pack.toml", "rb") as handle:
        manifest = tomllib.load(handle)["pack"]
    assert manifest["namespaces"] == ["partner"]
    assert manifest["presentation"]["display_name"] == "Partner Nodes"
    with pytest.raises(MissingApiKeyError, match="Unauthorized.*comfy-api-key"):
        execute(request_spec(), RuntimeContext(transport=FixtureTransport()), {"text": "x"})


@pytest.mark.parametrize(
    "adapter",
    (
        HttpSyncBinary("binary"),
        MultiStage("stages"),
        MultipartMap("multipart"),
    ),
)
def test_unimplemented_typed_adapter_fails_loudly(adapter: Adapter) -> None:
    with pytest.raises(NotImplementedError, match="not implemented in this partner slice"):
        execute(OpSpec((adapter,)), RuntimeContext(transport=FixtureTransport()))


def test_submit_poll_states_queued_attempts_timeout_and_absolute_header_policy() -> None:
    transport = FixtureTransport(
        FixtureResponse(payload={"polling_url": "https://poll.test/task"}),
        FixtureResponse(payload={"status": "Queued"}),
        FixtureResponse(payload={"status": "Running", "progress": 0.5}),
        FixtureResponse(payload={"status": "Ready", "result": {"sample": "ok"}}),
    )
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    spec = OpSpec(
        (
            HttpSyncJson("submit", "/proxy/bfl/run"),
            SubmitPoll(
                "poll",
                "submit",
                completed=("Ready",),
                failed=("Error",),
                queued=("Queued",),
                interval=5,
                max_attempts=2,
            ),
        )
    )
    execute(spec, RuntimeContext(transport=transport, api_key="key", sleep=sleep))
    assert sleeps == [5, 5]
    assert [request[2] for request in transport.requests[1:]] == [
        {"Accept": "application/json"},
        {"Accept": "application/json"},
        {"Accept": "application/json"},
    ]

    failed = FixtureTransport(
        FixtureResponse(payload={"polling_url": "https://poll.test/task"}),
        FixtureResponse(payload={"status": "Error"}),
    )
    with pytest.raises(PartnerError, match="failed"):
        execute(spec, RuntimeContext(transport=failed, api_key="key", sleep=sleep))

    timeout = FixtureTransport(
        FixtureResponse(payload={"polling_url": "https://poll.test/task"}),
        FixtureResponse(payload={"status": "Running"}),
        FixtureResponse(payload={"status": "Running"}),
    )
    with pytest.raises(PartnerError, match="timed out"):
        execute(spec, RuntimeContext(transport=timeout, api_key="key", sleep=sleep))

    private = FixtureTransport(FixtureResponse(payload={"polling_url": "https://bad.test/x"}))
    private.resolutions["bad.test"] = ("127.0.0.1",)
    with pytest.raises(TrustPolicyError, match="bad.test"):
        execute(spec, RuntimeContext(transport=private, api_key="key", sleep=sleep))


def test_submit_poll_cancellation_invokes_declared_cancel_endpoint() -> None:
    poll_response = FixtureResponse(payload={"status": "Running"})
    cancel_response = FixtureResponse(payload={})
    transport = FixtureTransport(
        FixtureResponse(payload={"polling_url": "https://poll.test/task"}),
        poll_response,
        cancel_response,
    )
    cancelled = False
    request_count = 0

    def cancel_during_poll() -> None:
        nonlocal cancelled, request_count
        request_count += 1
        if request_count == 2:
            cancelled = True

    transport.on_request = cancel_during_poll
    spec = OpSpec(
        (
            HttpSyncJson("submit", "/proxy/bfl/run"),
            SubmitPoll("poll", "submit", cancel_path="/proxy/bfl/cancel"),
        )
    )
    with pytest.raises(OperationCancelled):
        execute(
            spec,
            RuntimeContext(
                transport=transport,
                api_key="key",
                cancelled=lambda: cancelled,
            ),
        )
    assert poll_response.closed
    assert transport.requests[-1][0:2] == (
        "POST",
        "https://api.comfy.org/proxy/bfl/cancel",
    )
    assert transport.requests[-1][2]["X-API-KEY"] == "key"
    assert cancel_response.closed


def test_proxy_upload_strips_credentials_allows_signed_port_and_closes_on_interrupt() -> None:
    allocation = FixtureResponse(
        payload={
            "upload_url": "https://storage.test:8443/signed",
            "download_url": "https://cdn.test/file.png",
        }
    )
    uploaded = FixtureResponse()
    transport = FixtureTransport(allocation, uploaded)
    spec = OpSpec((ProxyUpload("upload", "content", "x.png", "image/png"),))
    execute(spec, RuntimeContext(transport=transport, api_key="secret"), {"content": b"abc"})
    assert transport.requests[0][0:2] == (
        "POST",
        "https://api.comfy.org/customers/storage",
    )
    assert transport.requests[0][2]["X-API-KEY"] == "secret"
    assert transport.byte_requests == [
        ("PUT", "https://storage.test:8443/signed", {"Content-Type": "image/png"}, b"abc")
    ]
    assert uploaded.closed

    interrupted_response = FixtureResponse()
    interrupted = FixtureTransport(
        FixtureResponse(
            payload={
                "upload_url": "https://storage.test/signed",
                "download_url": "https://cdn.test/file.png",
            }
        ),
        interrupted_response,
    )
    cancelled = False

    def interrupt_put() -> None:
        nonlocal cancelled
        cancelled = True

    interrupted.on_byte_request = interrupt_put
    with pytest.raises(OperationCancelled):
        execute(
            spec,
            RuntimeContext(transport=interrupted, api_key="secret", cancelled=lambda: cancelled),
            {"content": b"abc"},
        )
    assert interrupted_response.closed


def test_proxy_upload_retries_transient_put_without_credentials() -> None:
    allocation = FixtureResponse(
        payload={
            "upload_url": "https://storage.test/signed",
            "download_url": "https://cdn.test/file.png",
        }
    )
    transport = FixtureTransport(
        allocation,
        FixtureResponse(503),
        TransportNetworkError("reset"),
        FixtureResponse(),
    )
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = execute(
        OpSpec((ProxyUpload("upload", "content", "x.png", "image/png"),)),
        RuntimeContext(transport=transport, api_key="secret", sleep=sleep),
        {"content": b"abc"},
    )
    assert result == {}
    assert sleeps == [1.0, 2.0]
    assert len(transport.byte_requests) == 3
    assert all(request[2] == {"Content-Type": "image/png"} for request in transport.byte_requests)


def test_media_constraints_mask_encode_and_download_decode() -> None:
    too_small = FixtureTransport()
    constrained = OpSpec(
        (
            MediaConstraints("validate", "image", min_width=256, min_height=256),
            HttpSyncJson("request", "/proxy/bfl/run"),
        )
    )
    with pytest.raises(PartnerError, match="width must be at least 256"):
        execute(
            constrained,
            RuntimeContext(transport=too_small, api_key="key"),
            {"image": np.zeros((1, 32, 32, 3), dtype=np.float32)},
        )
    assert too_small.requests == []

    bad_ratio = FixtureTransport()
    ratio_spec = OpSpec(
        (
            MediaConstraints("validate", "aspect_ratio", min_aspect_ratio=0.25, max_aspect_ratio=4),
            HttpSyncJson("request", "/proxy/bfl/run"),
        )
    )
    with pytest.raises(PartnerError, match="aspect ratio"):
        execute(
            ratio_spec,
            RuntimeContext(transport=bad_ratio, api_key="key"),
            {"aspect_ratio": "10:1"},
        )
    assert bad_ratio.requests == []

    request_response = FixtureResponse(payload={})
    prepared = FixtureTransport(request_response)
    media_spec = OpSpec(
        (
            MaskPrepare("prepared", "mask", "image"),
            EncodeMedia("encoded", "prepared"),
            HttpSyncJson(
                "request",
                "/proxy/bfl/run",
                body=(InputBinding("mask", "encoded"),),
            ),
        )
    )
    execute(
        media_spec,
        RuntimeContext(transport=prepared, api_key="key"),
        {
            "image": np.zeros((1, 4, 6, 3), dtype=np.float32),
            "mask": np.array(
                [
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                    [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        },
    )
    encoded = prepared.requests[0][3]["mask"]
    assert isinstance(encoded, str)
    decoded_mask = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded_mask.size == (6, 4)
    assert decoded_mask.mode == "RGB"
    pixels = np.asarray(decoded_mask)
    assert pixels[0, 0, 0] == 0
    assert pixels[0, 3, 0] == 255

    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(stream, "PNG")
    download = FixtureTransport(
        FixtureResponse(payload={"url": "https://media.test/image.png"}),
        FixtureResponse(headers={"Content-Type": "image/png"}, chunks=(stream.getvalue(),)),
    )
    result = execute(
        OpSpec(
            (
                HttpSyncJson("request", "/proxy/bfl/result"),
                DownloadDecode("decode", "request", ("url",), "image", byte_cap=1024),
            )
        ),
        RuntimeContext(transport=download, api_key="key"),
    )
    image = np.asarray(result["image"])
    assert image.shape == (1, 2, 3, 3)
    assert image.dtype == np.float32


def test_mask_prepare_preserves_nonuniform_batches_and_nearest_pixels() -> None:
    transport = FixtureTransport(FixtureResponse(payload={}))
    source = np.array(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    execute(
        OpSpec(
            (
                MaskPrepare("prepared", "mask", "image"),
                EncodeMedia("encoded", "prepared", batch_targets=("mask_1", "mask_2")),
                HttpSyncJson(
                    "request",
                    "/proxy/bfl/run",
                    body=(InputBinding("", "encoded", expand=True),),
                ),
            )
        ),
        RuntimeContext(transport=transport, api_key="key"),
        {
            "image": np.zeros((1, 4, 6, 3), dtype=np.float32),
            "mask": source,
        },
    )
    body = transport.requests[0][3]
    decoded = [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(body[name]))))
        for name in ("mask_1", "mask_2")
    ]
    expected = [
        np.repeat(np.repeat(item, 2, axis=0), 2, axis=1).astype(np.uint8) * 255 for item in source
    ]
    for actual, channel in zip(decoded, expected, strict=True):
        assert actual.shape == (4, 6, 3)
        assert np.array_equal(actual[..., 0], channel)
        assert np.array_equal(actual[..., 1], channel)
        assert np.array_equal(actual[..., 2], channel)


def test_encode_media_matches_bfl_four_megapixel_even_dimension_limit() -> None:
    response = FixtureResponse(payload={})
    transport = FixtureTransport(response)
    image = np.zeros((1, 2001, 3001, 3), dtype=np.float32)
    execute(
        OpSpec(
            (
                EncodeMedia("encoded", "image", max_pixels=2048 * 2048),
                HttpSyncJson(
                    "request",
                    "/proxy/bfl/run",
                    body=(InputBinding("image", "encoded"),),
                ),
            )
        ),
        RuntimeContext(transport=transport, api_key="key"),
        {"image": image},
    )
    encoded = transport.requests[0][3]["image"]
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.size == (2508, 1672)


def test_in_flight_request_is_cancelled_on_local_interruption() -> None:
    started = asyncio.Event()
    cancelled = False

    class HangingTransport(FixtureTransport):
        async def request(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.was_cancelled = True
                raise

    transport = HangingTransport()
    transport.was_cancelled = False

    async def scenario() -> None:
        nonlocal cancelled
        operation = asyncio.create_task(
            run_op(
                request_spec(),
                {"text": "x"},
                RuntimeContext(
                    transport=transport,
                    api_key="key",
                    cancelled=lambda: cancelled,
                ),
            )
        )
        await started.wait()
        cancelled = True
        with pytest.raises(OperationCancelled):
            await operation

    asyncio.run(scenario())
    assert transport.was_cancelled


def test_in_flight_json_body_and_download_chunk_are_cancelled() -> None:
    async def run_json_case() -> None:
        cancelled = False
        started = asyncio.Event()

        class HangingJson(FixtureResponse):
            async def json(self) -> object:
                started.set()
                await asyncio.Event().wait()
                return {}

        transport = FixtureTransport(HangingJson())
        operation = asyncio.create_task(
            run_op(
                request_spec(),
                {"text": "x"},
                RuntimeContext(
                    transport=transport,
                    api_key="key",
                    cancelled=lambda: cancelled,
                ),
            )
        )
        await started.wait()
        cancelled = True
        with pytest.raises(OperationCancelled):
            await operation
        assert transport.results == []

    async def run_chunk_case() -> None:
        cancelled = False
        started = asyncio.Event()

        class HangingChunk(FixtureResponse):
            async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
                started.set()
                await asyncio.Event().wait()
                yield b""

        response = HangingChunk(headers={"Content-Type": "image/png"})
        transport = FixtureTransport(response)
        operation = asyncio.create_task(
            download_bytes(
                "https://media.test/file",
                RuntimeContext(transport=transport, cancelled=lambda: cancelled),
                media_family="image",
            )
        )
        await started.wait()
        cancelled = True
        with pytest.raises(OperationCancelled):
            await operation
        assert response.closed

    asyncio.run(run_json_case())
    asyncio.run(run_chunk_case())


def test_retry_after_seconds_http_date_and_cap() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert _retry_after("12", 1, now) == 12
    assert _retry_after("999", 1, now) == 150
    date = (now + timedelta(seconds=42)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert _retry_after(date, 1, now) == 42
    assert _retry_after("bad", 3, now) == 3


def test_retry_state_machine_honors_retry_after_and_separate_429_budget() -> None:
    transport = FixtureTransport(
        FixtureResponse(503, headers={"Retry-After": "4"}),
        FixtureResponse(429, headers={"Retry-After": "7"}),
        FixtureResponse(payload={"result": {"id": "ok"}}),
    )
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    spec = request_spec(max_retries=1, max_rate_limit_retries=1)
    assert HttpSyncJson("defaults", "/proxy/x").max_rate_limit_retries == 16
    assert execute(
        spec,
        RuntimeContext(transport=transport, api_key="key", sleep=sleep),
        {"text": "x"},
    ) == {"task_id": "ok"}
    assert sleeps == [4, 7]
    assert len(transport.requests) == 3


@pytest.mark.parametrize(
    ("status", "message"),
    ((401, "login"), (402, "credits"), (409, "support@comfy.org"), (429, "Rate Limit")),
)
def test_friendly_http_errors(status: int, message: str) -> None:
    transport = FixtureTransport(FixtureResponse(status, payload={}))
    with pytest.raises(PartnerError, match=message):
        execute(
            request_spec(max_rate_limit_retries=0),
            RuntimeContext(transport=transport, api_key="key"),
            {"text": "x"},
        )


@pytest.mark.parametrize(
    ("internet", "error"), ((False, LocalNetworkError), (True, ApiServerError))
)
def test_exhausted_network_error_distinguishes_connectivity(
    internet: bool, error: type[Exception]
) -> None:
    transport = FixtureTransport(TransportNetworkError("down"))
    transport.internet = internet
    with pytest.raises(error):
        execute(
            request_spec(),
            RuntimeContext(transport=transport, api_key="key"),
            {"text": "x"},
        )


def test_private_ip_is_denied_after_dns_and_port_rule_is_closed() -> None:
    private = FixtureTransport()
    private.resolutions["private.test"] = ("10.2.3.4",)
    with pytest.raises(TrustPolicyError, match="private.test.*private or local"):
        asyncio.run(
            download_bytes(
                "https://private.test/file", RuntimeContext(transport=private), media_family="image"
            )
        )
    with pytest.raises(TrustPolicyError, match="port.test.*port 443"):
        asyncio.run(
            download_bytes(
                "https://port.test:8443/file",
                RuntimeContext(transport=FixtureTransport()),
                media_family="image",
            )
        )
    proxy = FixtureTransport(
        FixtureResponse(headers={"Content-Type": "image/png"}, chunks=(b"ok",))
    )
    assert (
        asyncio.run(
            download_bytes(
                "https://port.test:8443/file",
                RuntimeContext(transport=proxy),
                media_family="image",
                from_proxy=True,
            )
        )
        == b"ok"
    )


def test_redirects_revalidate_strip_headers_and_cap_at_three_hops() -> None:
    responses = [
        FixtureResponse(302, headers={"Location": f"https://hop{i}.test/file"}) for i in range(1, 5)
    ]
    transport = FixtureTransport(*responses)
    with pytest.raises(TrustPolicyError, match="redirect limit"):
        asyncio.run(
            download_bytes(
                "https://start.test/file",
                RuntimeContext(transport=transport, api_key="secret"),
                media_family="image",
            )
        )
    assert len(transport.requests) == 4
    assert all(headers == {} for _, _, headers, _ in transport.requests)

    revalidate = FixtureTransport(FixtureResponse(302, headers={"Location": "https://bad.test/x"}))
    revalidate.resolutions["bad.test"] = ("127.0.0.1",)
    with pytest.raises(TrustPolicyError, match="bad.test"):
        asyncio.run(
            download_bytes(
                "https://good.test/file",
                RuntimeContext(transport=revalidate),
                media_family="image",
            )
        )


def test_download_enforces_streaming_byte_cap_and_content_type_family() -> None:
    oversized = FixtureTransport(
        FixtureResponse(headers={"Content-Type": "image/png"}, chunks=(b"abc", b"def"))
    )
    with pytest.raises(TrustPolicyError, match="byte cap"):
        asyncio.run(
            download_bytes(
                "https://media.test/file",
                RuntimeContext(transport=oversized),
                media_family="image",
                byte_cap=5,
            )
        )
    wrong_type = FixtureTransport(
        FixtureResponse(headers={"Content-Type": "text/html"}, chunks=(b"x",))
    )
    with pytest.raises(TrustPolicyError, match="media.test.*not image"):
        asyncio.run(
            download_bytes(
                "https://media.test/file",
                RuntimeContext(transport=wrong_type),
                media_family="image",
            )
        )


def test_cancellation_is_checked_around_transport_awaits() -> None:
    response = FixtureResponse(payload={"result": {"id": "late"}})
    transport = FixtureTransport(response)
    cancelled = False

    def cancel_during_request() -> None:
        nonlocal cancelled
        cancelled = True

    transport.on_request = cancel_during_request
    with pytest.raises(OperationCancelled):
        execute(
            request_spec(),
            RuntimeContext(
                transport=transport,
                api_key="key",
                cancelled=lambda: cancelled,
            ),
            {"text": "x"},
        )
    assert response.closed


def test_pack_import_boundary_is_dinkster_api_v1_and_own_modules_only() -> None:
    forbidden: list[str] = []
    for path in sorted((PACK_ROOT / "src" / "dinkster_nodes_partner").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith("dinkster") and name != "dinkster_api.v1":
                    forbidden.append(f"{path.name}: {name}")
    assert forbidden == []


def test_amendment_a_nested_conflicts_name_both_full_targets_both_directions() -> None:
    for fixed in (
        (FixedField("outer", 1), FixedField("outer.inner", 2)),
        (FixedField("outer.inner", 1), FixedField("outer", 2)),
        (FixedField("outer.inner", 1), FixedField("outer.inner", 2)),
    ):
        with pytest.raises(ValueError) as error:
            ValueConstruct("value", "out", fixed=fixed)
        assert "outer" in str(error.value) and "outer.inner" in str(error.value)


def test_amendment_a_http_descriptor_rejects_static_nested_conflict_at_build_time() -> None:
    with pytest.raises(ValueError, match="request body conflict") as error:
        HttpSyncJson(
            "request",
            "/x",
            body=(InputBinding("outer.inner", "x"),),
            fixed=(FixedField("outer", 1),),
        )
    assert "outer" in str(error.value) and "outer.inner" in str(error.value)


def test_value_construct_builds_pinned_kling_camera_control_and_round_trips() -> None:
    spec = OpSpec(
        (
            ValueConstruct(
                "camera",
                "camera_control",
                (
                    InputBinding(
                        "config.horizontal",
                        "movement",
                        source_path=("horizontal",),
                    ),
                    InputBinding("config.vertical", "vertical_movement"),
                    InputBinding("config.pan", "pan", round_digits=2),
                    InputBinding("config.tilt", "tilt"),
                    InputBinding("config.roll", "roll"),
                    InputBinding("config.zoom", "zoom"),
                ),
                (FixedField("type", "simple"),),
            ),
            HttpSyncJson(
                "request",
                "/proxy/kling/camera",
                body=(InputBinding("camera_control", "camera"),),
            ),
        )
    )
    assert OpSpec.from_json(spec.to_json()) == spec
    transport = FixtureTransport(FixtureResponse(payload={}))
    outputs = execute(
        spec,
        RuntimeContext(transport=transport, api_key="key"),
        {
            "movement": {"horizontal": 1.0},
            "vertical_movement": -2.0,
            "pan": 0.126,
            "tilt": 0.0,
            "roll": None,
            "zoom": 3.0,
        },
    )
    expected = {
        "camera_control": {
            "type": "simple",
            "config": {
                "horizontal": 1.0,
                "vertical": -2.0,
                "pan": 0.13,
                "tilt": 0.0,
                "zoom": 3.0,
            },
        }
    }
    assert outputs == expected
    assert transport.requests[0][3] == expected


def test_amendment_a_expand_uses_literal_top_level_keys_and_shared_collision_diagnostic() -> None:
    result = execute(
        OpSpec((ValueConstruct("value", "out", (InputBinding("", "mapping", expand=True),)),)),
        RuntimeContext(transport=FixtureTransport()),
        {"mapping": {"a.b": 1}},
    )
    assert result == {"out": {"a.b": 1}}
    with pytest.raises(PartnerError, match="'a.b'.*'a'"):
        execute(
            OpSpec(
                (
                    ValueConstruct(
                        "value",
                        "out",
                        (InputBinding("", "mapping", expand=True),),
                        (FixedField("a.b", 1),),
                    ),
                )
            ),
            RuntimeContext(transport=FixtureTransport()),
            {"mapping": {"a": 2}},
        )


@pytest.mark.parametrize("value", ("abc", "A-1", "a_b", "v1.2", "9", "a.-_"))
def test_amendment_b_submit_template_approved_regex_values(value: str) -> None:
    transport = FixtureTransport(FixtureResponse(payload={"status": "done"}))
    execute(
        OpSpec(
            (
                SubmitPoll(
                    "poll", "submitted", path_template="jobs/{value}", path_value_path=("id",)
                ),
            )
        ),
        RuntimeContext(transport=transport, api_key="key", sleep=lambda _: asyncio.sleep(0)),
        {"submitted": {"id": value}},
    )
    assert transport.requests[0][1].endswith(f"/jobs/{value}")


@pytest.mark.parametrize("value", (".", "..", "---", "___", "a/b", "a b", "a?b"))
def test_amendment_b_submit_template_rejects_unsafe_values_before_network(value: str) -> None:
    transport = FixtureTransport()
    with pytest.raises(PartnerError, match="unsafe"):
        execute(
            OpSpec(
                (
                    SubmitPoll(
                        "poll", "submitted", path_template="jobs/{value}", path_value_path=("id",)
                    ),
                )
            ),
            RuntimeContext(transport=transport, api_key="key"),
            {"submitted": {"id": value}},
        )
    assert transport.requests == []


@pytest.mark.parametrize("template", ("//host/{value}", "https://host/{value}", "\\host\\{value}"))
def test_amendment_b_submit_template_constructor_rejects_absolute_network(template: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        SubmitPoll("poll", path_template=template, path_value_path=("id",))


def test_amendment_b_submit_template_exact_default_and_pairing() -> None:
    assert SubmitPoll("poll").path_template == ""
    assert (
        SubmitPoll(
            "poll", path_template="/proxy/jobs/{value}", path_value_path=("id",)
        ).path_template
        == "/proxy/jobs/{value}"
    )
    for kwargs in ({"path_template": "jobs/{value}"}, {"path_value_path": ("id",)}):
        with pytest.raises(ValueError, match="both"):
            SubmitPoll("poll", **kwargs)


@pytest.mark.parametrize(
    ("family", "format", "prefix"),
    (
        ("image", "PNG", "data:image/png"),
        ("image", "JPEG", "data:image/jpeg"),
        ("video", "MP4", "data:video/mp4"),
        ("audio", "MP3", "data:audio/mpeg"),
    ),
)
def test_amendment_c_closed_data_url_table(family: str, format: str, prefix: str) -> None:
    transport = FixtureTransport(FixtureResponse())
    raw: object = b"payload"
    if family == "audio":
        raw = {
            "waveform": np.zeros((1, 1, 1600), dtype=np.float32),
            "sample_rate": 16000,
        }
    execute(
        OpSpec(
            (
                EncodeMedia("encode", "raw", family, format, "data_url"),
                HttpSyncJson("send", "/x", body=(InputBinding("x", "encode"),)),
            )
        ),
        RuntimeContext(transport=transport, api_key="key"),
        {"raw": raw},
    )
    assert transport.requests[0][3]["x"].startswith(prefix)


def test_amendment_c_audio_mp3_real_decode_uses_supplied_rate() -> None:
    import av

    rate = 16000
    waveform = np.zeros((1, 1, rate // 10), dtype=np.float32)
    transport = FixtureTransport(FixtureResponse())
    execute(
        OpSpec(
            (
                EncodeMedia("encode", "audio", "audio", "MP3", "bytes"),
                HttpSyncJson("send", "/x", body=(InputBinding("x", "encode"),)),
            )
        ),
        RuntimeContext(transport=transport, api_key="key"),
        {"audio": {"waveform": waveform, "sample_rate": rate}},
    )
    raw = transport.requests[0][3]["x"]
    with av.open(io.BytesIO(raw), format="mp3") as container:
        assert container.streams.audio[0].rate == rate
        assert sum(frame.samples for frame in container.decode(audio=0)) > 0


@pytest.mark.parametrize(
    "audio",
    (
        {},
        b"not-pinned-audio",
        {"waveform": np.zeros((1, 1, 2), np.float64), "sample_rate": 1},
        {"waveform": np.zeros((1, 3, 2), np.float32), "sample_rate": 1},
        {"waveform": np.zeros((1, 1, 2), np.float32), "sample_rate": True},
    ),
)
def test_amendment_c_audio_strict_defect_diagnostics(audio: object) -> None:
    with pytest.raises(PartnerError, match="audio"):
        execute(
            OpSpec((EncodeMedia("encode", "audio", "audio", "MP3", "bytes"),)),
            RuntimeContext(transport=FixtureTransport()),
            {"audio": audio},
        )


def test_amendment_c_video_passthrough_decode_magic_and_proxy_content_type() -> None:
    raw = b"\0\0\0\x18ftypisom" + b"video"
    transport = FixtureTransport(
        FixtureResponse(payload={"url": "https://media.test/x"}),
        FixtureResponse(headers={"Content-Type": "video/mp4"}, chunks=(raw,)),
        FixtureResponse(
            payload={"upload_url": "https://storage.test/x", "download_url": "https://cdn.test/x"}
        ),
        FixtureResponse(),
    )
    execute(
        OpSpec(
            (
                HttpSyncJson("request", "/result"),
                DownloadDecode("decode", "request", ("url",), "video", "video"),
                ProxyUpload("upload", "decode"),
            )
        ),
        RuntimeContext(transport=transport, api_key="key"),
    )
    assert transport.requests[2][3]["content_type"] == "video/mp4"
    assert transport.byte_requests[0][2] == {"Content-Type": "video/mp4"}
    assert transport.byte_requests[0][3] == raw


def test_amendment_c_multi_download_skips_absent_none_concatenates_and_empty_loud() -> None:
    streams = []
    for color in ((1, 2, 3), (4, 5, 6)):
        stream = io.BytesIO()
        Image.new("RGB", (2, 1), color).save(stream, "PNG")
        streams.append(stream.getvalue())
    step = DownloadDecode(
        "decode", "request", (), "images", "image", 1024, ("items",), ("result", "url")
    )
    transport = FixtureTransport(
        FixtureResponse(
            payload={
                "items": [
                    {"result": {"url": "https://a.test/x"}},
                    {},
                    None,
                    {"result": {"url": None}},
                    {"result": {"url": "https://b.test/x"}},
                ]
            }
        ),
        FixtureResponse(headers={"Content-Type": "image/png"}, chunks=(streams[0],)),
        FixtureResponse(headers={"Content-Type": "image/png"}, chunks=(streams[1],)),
    )
    result = execute(
        OpSpec((HttpSyncJson("request", "/result"), step)),
        RuntimeContext(transport=transport, api_key="key"),
    )
    assert np.asarray(result["images"]).shape == (2, 1, 2, 3)
    empty = FixtureTransport(FixtureResponse(payload={"items": [{}, None]}))
    with pytest.raises(PartnerError, match="no media"):
        execute(
            OpSpec((HttpSyncJson("request", "/result"), step)),
            RuntimeContext(transport=empty, api_key="key"),
        )


def test_audio_download_decode_remains_loudly_deferred() -> None:
    transport = FixtureTransport(
        FixtureResponse(payload={"url": "https://audio.test/result.mp3"}),
        FixtureResponse(headers={"Content-Type": "audio/mpeg"}, chunks=(b"mp3",)),
    )
    with pytest.raises(PartnerError, match="audio download decode is not implemented"):
        execute(
            OpSpec(
                (
                    HttpSyncJson("request", "/result"),
                    DownloadDecode("download", "request", ("url",), "audio", "audio"),
                )
            ),
            RuntimeContext(transport=transport, api_key="key"),
        )


def test_amendment_c_batch_join_validation_preserves_order_and_names_key() -> None:
    spec = OpSpec(
        (
            BatchMapJoin("join", "items", "url"),
            ValueConstruct("out", "out", (InputBinding("joined", "join"),)),
        )
    )
    assert execute(
        spec,
        RuntimeContext(transport=FixtureTransport()),
        {"items": {"b": "B", "a": None, "c": "C"}},
    )["out"]["joined"] == [{"url": "B"}, {"url": "C"}]
    for value, match in (({1: "x"}, "1"), ({"bad": 3}, "bad")):
        with pytest.raises(PartnerError, match=match):
            execute(
                OpSpec((BatchMapJoin("join", "items", "url"),)),
                RuntimeContext(transport=FixtureTransport()),
                {"items": value},
            )


def test_amendment_c_batch_upload_wraps_item_failure_without_partial_return() -> None:
    transport = FixtureTransport(
        FixtureResponse(payload={"upload_url": "bad", "download_url": "x"})
    )
    with pytest.raises(PartnerError, match="item 'second'"):
        execute(
            OpSpec((ProxyUpload("upload", "items", batch=True),)),
            RuntimeContext(transport=transport, api_key="key"),
            {"items": {"second": b"x"}},
        )


def test_amendment_d1_binding_precedence_and_lower_non_string_loud() -> None:
    binding = InputBinding(
        "x",
        "source",
        source_path=("value",),
        round_digits=1,
        present_if="enabled",
        string_case="lower",
        omit_if="NONE",
    )
    result = execute(
        OpSpec((ValueConstruct("value", "out", (binding,)),)),
        RuntimeContext(transport=FixtureTransport()),
        {"source": {"value": "NONE"}, "enabled": True},
    )
    assert result == {"out": {}}
    retained_null = execute(
        OpSpec(
            (
                ValueConstruct(
                    "value",
                    "out",
                    (InputBinding("x", "x", omit_none=False, omit_if="auto"),),
                ),
            )
        ),
        RuntimeContext(transport=FixtureTransport()),
        {"x": "auto"},
    )
    assert retained_null == {"out": {"x": None}}
    with pytest.raises(PartnerError, match="must be a string"):
        execute(
            OpSpec(
                (ValueConstruct("value", "out", (InputBinding("x", "x", string_case="lower"),)),)
            ),
            RuntimeContext(transport=FixtureTransport()),
            {"x": 3},
        )


@pytest.mark.parametrize(
    "field", ("source_path", "round_digits", "present_if", "string_case", "omit_if")
)
def test_amendment_d1_expand_rejects_every_incompatible_field(field: str) -> None:
    values = {
        "target": "",
        "input": "x",
        "expand": True,
        field: {
            "source_path": ("x",),
            "round_digits": 1,
            "present_if": "x",
            "string_case": "lower",
            "omit_if": "x",
        }[field],
    }
    with pytest.raises(ValueError, match="transforms or conditions"):
        InputBinding(**values)


@pytest.mark.parametrize(
    "op,value",
    (
        ("eq", None),
        ("ne", None),
        ("present", 1),
        ("absent", False),
        ("count_eq", True),
        ("count_le", -1),
        ("strip_min_len", 0),
    ),
)
def test_amendment_d2_cond_pairing_rejects_invalid_shapes(op: str, value: object) -> None:
    with pytest.raises(ValueError):
        Cond("x", op, value)


def test_amendment_d2_check_semantics_count_strip_and_zero_transport() -> None:
    transport = FixtureTransport()
    checks = CheckInputs(
        "checks",
        (Check("bad", require=(Cond("images", "count_eq", 3), Cond("text", "strip_min_len", 2))),),
    )
    with pytest.raises(PartnerError, match="bad"):
        execute(
            OpSpec((checks, HttpSyncJson("network", "/x"))),
            RuntimeContext(transport=transport, api_key="key"),
            {"images": {"a": np.zeros((2, 1, 1, 3)), "b": np.zeros((1, 1, 3))}, "text": " "},
        )
    assert transport.requests == []


def test_amendment_d2_eq_ne_missing_semantics_and_nonscalar_loud() -> None:
    checks = CheckInputs(
        "checks",
        (
            Check("eq", when=(Cond("missing", "eq", 1),), require=(Cond("ok", "present"),)),
            Check("ne", when=(Cond("missing", "ne", 1),), require=(Cond("ok", "present"),)),
        ),
    )
    with pytest.raises(PartnerError, match="ne"):
        execute(OpSpec((checks,)), RuntimeContext(transport=FixtureTransport()))
    with pytest.raises(PartnerError, match="scalar"):
        execute(
            OpSpec(
                (
                    CheckInputs(
                        "checks",
                        (Check("x", when=(Cond("x", "eq", 1),), require=(Cond("ok", "present"),)),),
                    ),
                )
            ),
            RuntimeContext(transport=FixtureTransport()),
            {"x": [1]},
        )


def test_amendment_d2_strict_value_construct_json_objects() -> None:
    data = json.loads(OpSpec((ValueConstruct("value", "out"),)).to_json())
    data["adapters"][0]["bindings"] = ["bad"]
    with pytest.raises(ValueError, match="objects"):
        OpSpec.from_json(json.dumps(data))


def test_amendment_d2_duration_probe_exact_mapping_bounds_and_image_max_bytes() -> None:
    import av

    stream = io.BytesIO()
    with av.open(stream, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=10)
        video.width = 16
        video.height = 16
        video.pix_fmt = "yuv420p"
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    pinned = {"container": "mp4", "bytes": stream.getvalue()}
    execute(
        OpSpec((MediaConstraints("duration", "video", min_duration=0.9999, max_duration=1.0001),)),
        RuntimeContext(transport=FixtureTransport()),
        {"video": pinned},
    )
    with pytest.raises(PartnerError, match="at most"):
        execute(
            OpSpec((MediaConstraints("duration", "video", max_duration=0.9),)),
            RuntimeContext(transport=FixtureTransport()),
            {"video": pinned},
        )
    execute(
        OpSpec((MediaConstraints("image", "image", max_bytes=100),)),
        RuntimeContext(transport=FixtureTransport()),
        {"image": np.zeros((1, 2, 2, 3), np.float32)},
    )


@pytest.mark.parametrize(
    "value", ({"container": "mp4"}, b"bad", {"container": "bad", "bytes": b"x"})
)
def test_amendment_d2_duration_malformed_or_unparseable_video_loud(value: object) -> None:
    with pytest.raises(PartnerError, match="video|probe"):
        execute(
            OpSpec((MediaConstraints("duration", "video", max_duration=1),)),
            RuntimeContext(transport=FixtureTransport()),
            {"video": value},
        )
