from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from aiohttp import web
from dinkster_api.v1 import (
    SCHEMA_WIRE_VERSION,
    AssetError,
    AssetRef,
    AssetVault,
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    digest_bytes,
    schema_signature,
    schema_to_wire,
)
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_remote import catalog as remote_catalog
from dinkster_nodes_remote.catalog import CatalogError, parse_catalog
from dinkster_nodes_remote.runtime import (
    GatewayError,
    RemoteConfig,
    RemoteRuntime,
    build_node_classes,
)
from dinkster_schema import use_reporter
from dinkster_workers import ExecutionContext, use_execution_context
from dinkster_workers.boundary import write_frame

IMAGE = TypeExpr.asset_of(TypeExpr.concrete("dinkster.image"))
STRING = TypeExpr.concrete("core.string")
RECORDED_GATEWAY_FIXTURE = Path(__file__).parent / "fixtures" / "gateway_session.json"


def remote_schema(node_type: str = "dinkster.remote.image", version: int = 1) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        version=version,
        display_name="Remote Image",
        category="remote",
        inputs=(InputSpec("prompt", STRING), InputSpec("source", IMAGE, required=False)),
        outputs=(OutputSpec("image", IMAGE),),
        idempotent=False,
        io_bound=True,
        emits_previews=True,
    )


def catalog_payload(
    *schemas: NodeSchema,
    epoch: int = 1,
    schema_wire: int = SCHEMA_WIRE_VERSION,
) -> dict[str, object]:
    nodes: dict[str, object] = {}
    for schema in schemas:
        nodes[schema.node_type] = {
            "nodeType": schema.node_type,
            "latestVersion": schema.version,
            "schemaVersions": {
                str(schema.version): {
                    "schema": schema_to_wire(schema, wire_version=schema_wire),
                    "signature": schema_signature(schema),
                }
            },
        }
    return {"catalogEpoch": epoch, "schemaWire": schema_wire, "nodes": nodes}


def config(base_url: str, cache: Path, *, token: str = "session-token") -> RemoteConfig:
    return RemoteConfig(
        catalog_base=base_url,
        gateway_base=base_url,
        cache_path=cache,
        token=token,
        token_file=None,
        catalog_poll_interval=0.02,
        image_poll_interval=0.01,
        video_poll_interval=0.02,
        request_timeout=1,
        max_retries=2,
        max_retry_after=0.1,
    )


class FakeGateway:
    def __init__(self, *schemas: NodeSchema, recording: Mapping[str, object] | None = None) -> None:
        self.schemas = schemas or (remote_schema(),)
        self.recording = recording
        self.epoch = 1
        self.catalog_etag = '"catalog-1"'
        self.epoch_etag = '"epoch-1"'
        self.catalog_status = 200
        self.submit_failures = 0
        self.submit_error: tuple[int, str, bool, str | None] | None = None
        self.run_forever = False
        self.output_bytes = b"verified remote image"
        self.preview_bytes = b"preview"
        self.output_digest: str | None = None
        self.submissions: list[dict[str, object]] = []
        self.idempotency_keys: list[str] = []
        self.uploads: list[tuple[str | None, bytes]] = []
        self.catalog_headers: list[str | None] = []
        self.epoch_headers: list[str | None] = []
        self.download_authorizations: list[str | None] = []
        self.cancelled = asyncio.Event()
        self.submitted = asyncio.Event()
        self.polled = asyncio.Event()
        self._runner: web.AppRunner | None = None
        self.base_url = ""

    async def __aenter__(self) -> FakeGateway:
        app = web.Application()
        app.router.add_get("/catalog/nodes", self._catalog)
        app.router.add_get("/catalog/epoch", self._epoch)
        app.router.add_post("/v1/uploads", self._upload)
        app.router.add_post("/v1/jobs", self._submit)
        app.router.add_get("/v1/jobs/{job_id}", self._poll)
        app.router.add_delete("/v1/jobs/{job_id}", self._cancel)
        app.router.add_get("/download/{kind}", self._download)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        assert self._runner.addresses
        port = int(self._runner.addresses[0][1])
        self.base_url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *_args: object) -> None:
        assert self._runner is not None
        await self._runner.cleanup()

    def replace_catalog(self, *schemas: NodeSchema) -> None:
        self.schemas = schemas
        self.epoch += 1
        self.catalog_etag = f'"catalog-{self.epoch}"'
        self.epoch_etag = f'"epoch-{self.epoch}"'

    def _authorized(self, request: web.Request) -> web.Response | None:
        if request.headers.get("Authorization") == "Bearer session-token":
            return None
        return web.json_response(
            {
                "error": {
                    "code": "unauthenticated",
                    "source": "gateway",
                    "retryable": False,
                    "message": "session required",
                }
            },
            status=401,
        )

    async def _catalog(self, request: web.Request) -> web.Response:
        self.catalog_headers.append(request.headers.get("If-None-Match"))
        assert "wire" not in request.query
        if self.catalog_status != 200:
            return web.Response(status=self.catalog_status)
        if request.headers.get("If-None-Match") == self.catalog_etag:
            return web.Response(status=304)
        if self.recording is not None:
            return web.json_response(
                cast("dict[str, object]", self.recording["catalog"]),
                headers={"ETag": self.catalog_etag},
            )
        return web.json_response(
            catalog_payload(*self.schemas, epoch=self.epoch),
            headers={"ETag": self.catalog_etag},
        )

    async def _epoch(self, request: web.Request) -> web.Response:
        self.epoch_headers.append(request.headers.get("If-None-Match"))
        if request.headers.get("If-None-Match") == self.epoch_etag:
            return web.Response(status=304)
        return web.json_response({"catalogEpoch": self.epoch}, headers={"ETag": self.epoch_etag})

    async def _upload(self, request: web.Request) -> web.Response:
        refused = self._authorized(request)
        if refused is not None:
            return refused
        body = await request.read()
        digest = request.headers.get("X-Dinkster-Digest")
        self.uploads.append((digest, body))
        return web.json_response({"digest": digest}, status=201)

    def _descriptor(self, kind: str, data: bytes) -> dict[str, object]:
        digest = (
            self.output_digest if kind == "output" and self.output_digest else digest_bytes(data)
        )
        return {
            "digest": digest,
            "size": len(data),
            "name": f"{kind}.bin",
            "mediaType": "image/png",
            "downloadUrl": f"{self.base_url}/download/{kind}",
            "width": 2,
            "height": 1,
            "stream": "main" if kind == "preview" else None,
            "frameIndex": 0 if kind == "preview" else None,
            "frameCount": 4 if kind == "preview" else None,
            "fps": 12.5 if kind == "preview" else None,
        }

    async def _submit(self, request: web.Request) -> web.Response:
        refused = self._authorized(request)
        if refused is not None:
            return refused
        body = cast("dict[str, object]", await request.json())
        self.submissions.append(body)
        self.idempotency_keys.append(request.headers.get("Idempotency-Key", ""))
        if self.recording is not None:
            return web.json_response(
                cast("dict[str, object]", self.recording["submit"]), status=202
            )
        if self.submit_failures:
            self.submit_failures -= 1
            return web.json_response(
                {
                    "error": {
                        "code": "partner_unavailable",
                        "source": "gateway",
                        "retryable": True,
                        "message": "temporary upstream outage",
                    }
                },
                status=503,
                headers={"Retry-After": "0"},
            )
        if self.submit_error is not None:
            status, code, retryable, retry_after = self.submit_error
            headers = {"Retry-After": retry_after} if retry_after is not None else None
            return web.json_response(
                {
                    "error": {
                        "code": code,
                        "source": "gateway",
                        "retryable": retryable,
                        "message": f"fake {code}",
                    }
                },
                status=status,
                headers=headers,
            )
        self.submitted.set()
        return web.json_response(
            {
                "job": {
                    "jobId": "job-1",
                    "state": "running",
                    "progress": {"step": 1, "total": 2, "text": "remote"},
                    "preview": self._descriptor("preview", self.preview_bytes),
                }
            },
            status=202,
        )

    async def _poll(self, request: web.Request) -> web.Response:
        refused = self._authorized(request)
        if refused is not None:
            return refused
        self.polled.set()
        if self.recording is not None:
            encoded = json.dumps(self.recording["poll"]).replace("{base_url}", self.base_url)
            return web.json_response(cast("dict[str, object]", json.loads(encoded)))
        if self.run_forever:
            return web.json_response({"jobId": "job-1", "state": "running"})
        return web.json_response(
            {
                "jobId": "job-1",
                "state": "succeeded",
                "progress": {"step": 2, "total": 2},
                "outputs": {"image": self._descriptor("output", self.output_bytes)},
            }
        )

    async def _cancel(self, request: web.Request) -> web.Response:
        refused = self._authorized(request)
        if refused is not None:
            return refused
        self.cancelled.set()
        return web.json_response({"jobId": request.match_info["job_id"], "state": "cancelled"})

    async def _download(self, request: web.Request) -> web.Response:
        self.download_authorizations.append(request.headers.get("Authorization"))
        kind = request.match_info["kind"]
        if self.recording is None:
            data = self.preview_bytes if kind == "preview" else self.output_bytes
        else:
            assets = cast("Mapping[str, object]", self.recording["assets"])
            data = base64.b64decode(cast("str", assets[kind]))
        return web.Response(body=data, content_type="image/png")


def input_asset(root: Path, data: bytes = b"input image") -> AssetRef:
    vault = AssetVault(root)
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return AssetRef(digest, "input.png", len(data), "image/png", resolver=vault)


def test_catalog_enforces_remote_schema_invariants_and_isolates_bad_entries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    valid = remote_schema()
    payload = catalog_payload(valid)
    nodes = cast("dict[str, object]", payload["nodes"])
    nodes["outside.remote"] = {
        "nodeType": "outside.remote",
        "latestVersion": 1,
        "schemaVersions": {"1": {"schema": {}, "signature": "bad"}},
    }

    snapshot = parse_catalog(payload)

    assert [entry.schema.node_type for entry in snapshot.schemas] == [valid.node_type]
    assert "skipping malformed remote catalog entry" in caplog.text
    for invalid in (
        replace(valid, io_bound=False),
        replace(valid, idempotent=True),
        replace(valid, io_bound=False, occupies=("gpu",)),
        replace(valid, node_type="other.remote.image"),
    ):
        assert parse_catalog(catalog_payload(invalid)).schemas == ()
    snapshot = parse_catalog(catalog_payload(valid))
    assert snapshot.schema_wire == SCHEMA_WIRE_VERSION
    wrong_entry_wire = catalog_payload(valid)
    wrong_nodes = cast("dict[str, object]", wrong_entry_wire["nodes"])
    wrong_entry = cast("dict[str, object]", wrong_nodes[valid.node_type])
    wrong_versions = cast("dict[str, object]", wrong_entry["schemaVersions"])
    wrong_version = cast("dict[str, object]", wrong_versions["1"])
    wrong_version["schema"] = {**schema_to_wire(valid), "schemaVersion": 2}
    assert parse_catalog(wrong_entry_wire).schemas == ()
    with pytest.raises(CatalogError, match="unsupported schema wire"):
        parse_catalog({**catalog_payload(valid), "schemaWire": 2})


def test_catalog_rejects_entry_schema_wire_mismatch() -> None:
    valid = remote_schema()
    payload = catalog_payload(valid)
    nodes = cast("dict[str, object]", payload["nodes"])
    parse_entry = cast(
        "Callable[[str, object, int], object]",
        vars(remote_catalog)["_remote_schema"],
    )

    with pytest.raises(CatalogError, match="does not use the catalog's schema wire version"):
        parse_entry(valid.node_type, nodes[valid.node_type], SCHEMA_WIRE_VERSION + 1)


def test_dynamic_node_classes_keep_their_own_catalog_schema(tmp_path: Path) -> None:
    runtime = RemoteRuntime(config("", tmp_path / "unused"))
    runtime.snapshot = parse_catalog(
        catalog_payload(remote_schema("dinkster.remote.one"), remote_schema("dinkster.remote.two"))
    )

    nodes = build_node_classes(runtime)

    assert [node.define_schema().node_type for node in nodes] == [
        "dinkster.remote.one",
        "dinkster.remote.two",
    ]


def test_catalog_cache_degrades_startup_and_epoch_reload_uses_etags(tmp_path: Path) -> None:
    async def scenario() -> None:
        cache = tmp_path / "catalog.json"
        async with FakeGateway() as gateway:
            runtime = await asyncio.to_thread(RemoteRuntime, config(gateway.base_url, cache))
            assert runtime.snapshot.epoch == 1
            assert cache.is_file()

            gateway.replace_catalog(remote_schema(version=2))
            await asyncio.wait_for(runtime.wait_for_schema_reload(), 2)
            assert runtime.catalog.snapshot.epoch == 2
            assert runtime.catalog.snapshot.schemas[0].schema.version == 2
            assert gateway.catalog_headers[-1] == '"catalog-1"'
            assert gateway.epoch_headers[-1] is None
            cached = cache.read_bytes()

            gateway.epoch = 1
            gateway.schemas = (remote_schema(),)
            gateway.catalog_etag = '"stale"'
            assert await runtime.catalog.refresh(conditional=False) is None
            assert runtime.catalog.snapshot.epoch == 2
            assert cache.read_bytes() == cached

            gateway.epoch = 2
            gateway.schemas = (remote_schema(version=3),)
            gateway.catalog_etag = '"equivocating"'
            assert await runtime.catalog.refresh(conditional=False) is None
            assert runtime.catalog.snapshot.schemas[0].schema.version == 2
            assert cache.read_bytes() == cached

            gateway.catalog_status = 503
            fallback = await asyncio.to_thread(RemoteRuntime, config(gateway.base_url, cache))
            assert fallback.snapshot.epoch == 2
            assert fallback.snapshot.schemas[0].schema.version == 2

        empty = await asyncio.to_thread(
            RemoteRuntime,
            replace(config("http://127.0.0.1:9", tmp_path / "missing"), request_timeout=0.05),
        )
        assert empty.snapshot.schemas == ()

    asyncio.run(scenario())


def test_recorded_gateway_catalog_submit_progress_and_digest_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = cast(
        "dict[str, object]", json.loads(RECORDED_GATEWAY_FIXTURE.read_text(encoding="utf-8"))
    )
    assert recording["sourceCommit"] == "3bbf143bcd193ab508d7bb409b3dc61522d3f412"

    async def scenario() -> None:
        async with FakeGateway(recording=recording) as gateway:
            runtime = await asyncio.to_thread(
                RemoteRuntime, config(gateway.base_url, tmp_path / "catalog.json")
            )
            assert [entry.schema.node_type for entry in runtime.snapshot.schemas] == [
                "dinkster.remote.nanobanana.image",
                "dinkster.remote.seedance.video",
            ]
            vault = tmp_path / "output-vault"
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault))
            reports: list[tuple[str, Mapping[str, object], bytes | None]] = []
            with use_reporter(lambda name, data, blob: reports.append((name, data, blob))):
                result = await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})

            assert len(gateway.submissions) == 1
            assert gateway.submissions[0]["nodeType"] == "dinkster.remote.nanobanana.image"
            assert gateway.submissions[0]["inputs"] == {"prompt": "hello"}
            assert [entry[0] for entry in reports] == ["progress", "progress"]
            assert [entry[1]["step"] for entry in reports] == [1, 2]
            output = result["image"]
            assert isinstance(output, AssetRef)
            assets = cast("Mapping[str, object]", recording["assets"])
            assert output.read_bytes() == base64.b64decode(cast("str", assets["output"]))
            assert output.digest == digest_bytes(output.read_bytes())
            assert AssetVault(vault).has(output.digest)

    asyncio.run(scenario())


def test_upload_submit_poll_preview_and_verified_vault_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with FakeGateway() as gateway:
            gateway.submit_failures = 1
            runtime = await asyncio.to_thread(
                RemoteRuntime, config(gateway.base_url, tmp_path / "catalog.json")
            )
            vault = tmp_path / "output-vault"
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault))
            source = input_asset(tmp_path / "input-vault")
            reports: list[tuple[str, Mapping[str, object], bytes | None]] = []
            with use_reporter(lambda name, data, blob: reports.append((name, data, blob))):
                result = await runtime.invoke(
                    runtime.snapshot.schemas[0], {"prompt": "hello", "source": source}
                )

            assert gateway.uploads == [(source.digest, b"input image")]
            assert len(gateway.submissions) == 2
            assert len(set(gateway.idempotency_keys)) == 1
            uuid.UUID(gateway.idempotency_keys[0])
            submission = gateway.submissions[-1]
            assert submission == {
                "nodeType": "dinkster.remote.image",
                "schemaVersion": 1,
                "schemaSignature": schema_signature(remote_schema()),
                "inputs": {
                    "prompt": "hello",
                    "source": {
                        "digest": source.digest,
                        "name": "input.png",
                        "size": len(b"input image"),
                        "mediaType": "image/png",
                    },
                },
                "routing": {"mode": "pooled"},
                "resultDelivery": "client",
                "waitSeconds": 30,
            }
            output = result["image"]
            assert isinstance(output, AssetRef)
            assert output.read_bytes() == gateway.output_bytes
            assert AssetVault(vault).has(output.digest)
            assert gateway.download_authorizations == [None, None]
            assert [entry[0] for entry in reports] == ["progress", "preview", "progress"]
            assert reports[1][2] == gateway.preview_bytes
            assert reports[1][1] == {
                "mime": "image/png",
                "width": 2,
                "height": 1,
                "stream": "main",
                "frameIndex": 0,
                "frameCount": 4,
                "fps": 12.5,
            }

    asyncio.run(scenario())


def test_digest_mismatch_never_enters_the_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with FakeGateway() as gateway:
            gateway.output_digest = digest_bytes(b"different bytes")
            runtime = await asyncio.to_thread(
                RemoteRuntime, config(gateway.base_url, tmp_path / "catalog.json")
            )
            vault = tmp_path / "output-vault"
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault))
            with pytest.raises(AssetError, match="did not verify"):
                await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
            assert AssetVault(vault).digests() == []

    asyncio.run(scenario())


def test_gateway_errors_remain_differentiated_and_honor_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        codes = (
            "unauthenticated",
            "insufficient_credit",
            "policy_denied",
            "schema_version_unsupported",
            "schema_signature_mismatch",
            "node_paused",
            "inventory_exhausted",
            "partner_unavailable",
            "partner_failed",
            "rate_limited",
        )
        async with FakeGateway() as gateway:
            runtime = await asyncio.to_thread(
                RemoteRuntime,
                replace(config(gateway.base_url, tmp_path / "catalog.json"), max_retries=0),
            )
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
            for code in codes:
                gateway.submit_error = (400, code, False, None)
                with pytest.raises(GatewayError) as raised:
                    await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
                assert raised.value.code == code
                assert f"code={code}" in str(raised.value)

            gateway.submit_error = (429, "rate_limited", True, "9")
            with pytest.raises(GatewayError) as raised:
                await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
            assert raised.value.retry_after == 9
            assert "Retry after 9 seconds" in str(raised.value)

    asyncio.run(scenario())


def test_logged_out_invocation_fails_before_gateway_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with FakeGateway() as gateway:
            runtime = await asyncio.to_thread(
                RemoteRuntime,
                config(gateway.base_url, tmp_path / "catalog.json", token=""),
            )
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
            assert runtime.snapshot.schemas
            with pytest.raises(GatewayError, match="Sign in") as raised:
                await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
            assert raised.value.code == "unauthenticated"
            assert gateway.submissions == []

    asyncio.run(scenario())


def test_rotated_token_file_is_read_for_each_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with FakeGateway() as gateway:
            token_file = tmp_path / "token"
            token_file.write_text("", encoding="utf-8")
            runtime = await asyncio.to_thread(
                RemoteRuntime,
                replace(
                    config(gateway.base_url, tmp_path / "catalog.json", token="fallback"),
                    token_file=token_file,
                ),
            )
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
            with pytest.raises(GatewayError, match="token file is empty"):
                await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
            assert gateway.submissions == []
            token_file.write_text("session-token", encoding="utf-8")
            await runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
            assert len(gateway.submissions) == 1

    asyncio.run(scenario())


def test_remote_service_urls_require_https_outside_loopback() -> None:
    defaults = RemoteConfig.from_env({})
    assert defaults.catalog_base == ""
    assert defaults.gateway_base == ""
    assert defaults.request_timeout > 30
    assert RemoteRuntime(defaults).catalog.timeout == 5
    with pytest.raises(ValueError, match="must use HTTPS"):
        RemoteConfig.from_env(
            {
                "DINKSTER_REMOTE_CATALOG_BASE": "http://catalog.example.test",
                "DINKSTER_REMOTE_GATEWAY_BASE": "https://gateway.example.test",
            }
        )


@pytest.mark.parametrize(
    "download_url",
    [
        "http://127.0.0.1:8765/private",
        "https://127.0.0.2/private",
        "https://127.1/private",
        "https://[::ffff:127.0.0.1]/private",
        "https://2130706433/private",
        "https://worker.localhost/private",
    ],
)
def test_production_gateway_cannot_redirect_downloads_to_loopback(
    tmp_path: Path, download_url: str
) -> None:
    runtime = RemoteRuntime(
        replace(
            config("", tmp_path / "catalog.json"),
            gateway_base="https://gateway.example.test",
        )
    )
    descriptor = {
        "downloadUrl": download_url,
        "digest": digest_bytes(b"private"),
        "size": len(b"private"),
    }

    with pytest.raises(GatewayError, match="outside the configured test gateway"):
        runtime._download_identity(descriptor, "remote output")  # pyright: ignore[reportPrivateUsage]


def test_execution_context_cancellation_deletes_the_remote_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with FakeGateway() as gateway:
            gateway.run_forever = True
            runtime = await asyncio.to_thread(
                RemoteRuntime, config(gateway.base_url, tmp_path / "catalog.json")
            )
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
            cancel_requested = asyncio.Event()
            with use_execution_context(
                ExecutionContext("native", None, cancelled=cancel_requested.is_set)
            ):
                task = asyncio.create_task(
                    runtime.invoke(runtime.snapshot.schemas[0], {"prompt": "hello"})
                )
            await asyncio.wait_for(gateway.submitted.wait(), 1)
            await asyncio.wait_for(gateway.polled.wait(), 1)
            cancel_requested.set()
            with pytest.raises(RuntimeError, match="cancelled by the client"):
                await task
            await asyncio.wait_for(gateway.cancelled.wait(), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("explicit_url", [False, True])
def test_default_doctor_uses_configured_remote_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_url: bool
) -> None:
    from argparse import Namespace

    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog

    from dinkster import compose, manager, serve
    from dinkster.installer import InstallError

    spec = compose.default_pack_spec("dinkster-nodes-remote")

    def prepared(spec: compose.PackSpec, **kwargs: object) -> compose.PackSpec:
        return spec

    monkeypatch.setattr(compose, "default_pack_specs", lambda: (spec,))
    monkeypatch.setattr(serve, "_prepare_default_pack", prepared)

    async def scenario() -> None:
        async with FakeGateway() as gateway:
            monkeypatch.setenv(
                "DINKSTER_REMOTE_CATALOG_BASE", "" if explicit_url else gateway.base_url
            )
            args = Namespace(
                defaults=True,
                library_root="",
                accelerator="cpu",
                remote_catalog_base=gateway.base_url if explicit_url else None,
                remote_gateway_base=None,
            )
            # Authoring diagnostics still fail doctor without discarding valid schemas.
            with pytest.raises(InstallError, match="unhealthy packs"):
                await asyncio.to_thread(manager._cmd_doctor, args)  # pyright: ignore[reportPrivateUsage]
            catalog = read_catalog(load_manifest(spec.manifest))
            assert catalog is not None
            assert "dinkster.remote.image" in catalog.schemas
            assert gateway.catalog_headers

    asyncio.run(scenario())


@pytest.mark.parametrize("catalog_mode", ["live", "persisted", "stale"])
def test_isolated_worker_announces_catalog_epoch_and_composer_replaces_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    catalog_mode: str,
) -> None:
    async def scenario() -> None:
        from dinkster_workers import diagnose

        from dinkster.compose import PackSpec, ServingComposer, default_pack_spec

        async with FakeGateway() as gateway:
            token = tmp_path / "token"
            token.write_text("session-token", encoding="utf-8")
            monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
            requests: asyncio.Queue[str] = asyncio.Queue()
            composer = ServingComposer(
                worker_env={"DINKSTER_ASSET_VAULT": str(tmp_path / "vault")},
                pack_scratch_root=tmp_path / "scratch",
                on_schema_reload=requests.put_nowait,
            )
            spec = PackSpec(
                manifest=Path(__file__).parents[1] / "dinkster-pack.toml",
                trust_reserved=True,
                env={
                    "DINKSTER_REMOTE_CATALOG_BASE": gateway.base_url,
                    "DINKSTER_REMOTE_GATEWAY_BASE": gateway.base_url,
                    "DINKSTER_REMOTE_AUTH_TOKEN_FILE": str(token),
                    "DINKSTER_REMOTE_CATALOG_POLL_INTERVAL": "0.02",
                },
                asset_vault_write=True,
                require_catalog=catalog_mode != "live",
            )
            try:
                if spec.require_catalog:
                    await asyncio.to_thread(diagnose, spec.manifest, environment=spec.env)
                await composer.add_pack(default_pack_spec("dinkster-nodes-media-io"))
                initial = await composer.add_pack(spec)
                assert "dinkster.remote.image" in composer.composition.schemas
                assert initial.execution_arms == {"dinkster.remote.image": ("native",)}
                if spec.require_catalog:
                    record = composer._records["dinkster-nodes-remote"]  # pyright: ignore[reportPrivateUsage]
                    assert record.worker.cold
                if catalog_mode == "stale":
                    gateway.replace_catalog(remote_schema(), remote_schema("dinkster.remote.extra"))
                engine = composer.composition.make_engine(lambda _event: None)
                result = await engine.run(
                    Graph(
                        nodes={
                            "remote": GraphNode(
                                "dinkster.remote.image", {"prompt": "through the boundary"}
                            )
                        }
                    ),
                    ["remote"],
                )
                output = result.outputs["remote"]["image"].resolve()
                assert isinstance(output, AssetRef)
                assert output.read_bytes() == gateway.output_bytes
                record = composer._records["dinkster-nodes-remote"]  # pyright: ignore[reportPrivateUsage]
                token_before = record.worker.instance_token
                assert token_before is not None
                if catalog_mode == "stale":
                    assert await asyncio.wait_for(requests.get(), 3) == "dinkster-nodes-remote"
                    live = await record.worker.ensure_started()
                    await composer.reload_pack("dinkster-nodes-remote")
                    current = composer._records["dinkster-nodes-remote"]  # pyright: ignore[reportPrivateUsage]
                    assert current.worker is live
                    assert live.instance_token == token_before
                    assert "dinkster.remote.extra" in composer.composition.schemas
                composer._schema_reload_requested(  # pyright: ignore[reportPrivateUsage]
                    "dinkster-nodes-remote", "not-active"
                )
                assert requests.empty()

                gateway.replace_catalog(remote_schema("dinkster.remote.video", version=2))
                assert await asyncio.wait_for(requests.get(), 3) == "dinkster-nodes-remote"
                result = await composer.reload_pack("dinkster-nodes-remote")
                assert "dinkster.remote.image" in result.removed_types
                assert set(result.delta.schemas) == {"dinkster.remote.video"}
                assert result.delta.execution_arms == {"dinkster.remote.video": ("native",)}
                assert "dinkster.remote.image" not in composer.composition.schemas
                assert "dinkster.remote.video" in composer.composition.schemas
                current = composer._records[  # pyright: ignore[reportPrivateUsage]
                    "dinkster-nodes-remote"
                ]
                assert current.worker.instance_token != token_before
            finally:
                await composer.close()

    asyncio.run(scenario())


@contextlib.asynccontextmanager
async def schema_reload_peer(
    *,
    capability: bool,
) -> AsyncGenerator[tuple[asyncio.StreamReader, asyncio.StreamWriter, asyncio.StreamWriter]]:
    peer_writer: asyncio.Future[asyncio.StreamWriter] = asyncio.get_running_loop().create_future()

    async def peer(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer_writer.set_result(writer)
        header: dict[str, object] = {
            "type": "hello",
            "pack": "remote-test",
            "workerInstance": "instance-1",
            "schemas": {},
        }
        if capability:
            header["schemaReload"] = True
        await write_frame(writer, header, [])

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    try:
        assert server.sockets
        host, port = server.sockets[0].getsockname()[:2]
        reader, writer = await asyncio.open_connection(host, port)
        yield reader, writer, await peer_writer
    finally:
        server.close()
        await server.wait_closed()


def test_boundary_accepts_schema_reload_only_after_hello_negotiation() -> None:
    async def scenario() -> None:
        from dinkster_values import TypeRegistry, register_core_types
        from dinkster_workers.boundary import ValueCodec
        from dinkster_workers.session import BoundarySession

        for capability in (True, False):
            called = asyncio.Event()
            registry = TypeRegistry()
            register_core_types(registry)
            async with schema_reload_peer(capability=capability) as (reader, writer, peer):
                session = BoundarySession(
                    registry,
                    role="test",
                    pack="remote-test",
                    codec=ValueCodec(registry),
                    on_schema_reload=lambda pack, token, event=called: event.set(),
                )
                try:
                    await session.begin(reader, writer, timeout=1)
                    await write_frame(peer, {"type": "schemaReloadRequest"}, [])
                    if capability:
                        await asyncio.wait_for(called.wait(), 1)
                        assert session.alive
                    else:
                        await asyncio.sleep(0.05)
                        assert not called.is_set()
                        assert not session.alive
                finally:
                    await session.close()
                    peer.close()
                    await peer.wait_closed()

    asyncio.run(scenario())
