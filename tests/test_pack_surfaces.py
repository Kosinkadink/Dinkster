"""Pack HTTP dispatch and module reads exercise real catalog and worker boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import (
    ActiveExtension,
    ExtensionSnapshot,
    JsonField,
    JsonObjectSchema,
    PackRoute,
)
from dinkster_protocol.frontend_modules import FRONTEND_CONTRIBUTION_KINDS
from dinkster_server import Principal, StaticBearerAuthenticator, create_app
from dinkster_server.auth import install_auth
from dinkster_server.pack_surfaces import install_pack_surfaces
from dinkster_workers import diagnose, load_manifest
from dinkster_workers.catalog import read_catalog, worker_declarations

from dinkster.compose import PackSpec, ServingComposer
from dinkster.extension_assets import read_module, resolve_frontend_modules

PROOF_ROOT = Path(__file__).resolve().parents[1] / "packages/dinkster-video/preview"
PROOF_PACK = "dinkster-video-preview"
PROOF_EVENT = "video-preview.initialized"
PROOF_ROUTE = f"/api/extensions/{PROOF_PACK}/routes/preview-policy"
SAFE_JSON_INTS = (-(2**53 - 1), 2**53 - 1)
UNSAFE_JSON_INTS = (-(2**53 + 1), -(2**53), 2**53, 2**53 + 1)


def test_frontend_contribution_vocabulary_is_the_supported_set() -> None:
    assert FRONTEND_CONTRIBUTION_KINDS == (
        "widgetKind",
        "widgetView",
        "previewRenderer",
        "textEditorExtension",
        "menu",
        "command",
        "keybinding",
        "setting",
        "canvasLayer",
        "nodeDecoration",
        "hostUi",
        "searchProvider",
        "workflowObserver",
        "eventConsumer",
        "workflowImporter",
    )


@pytest.fixture
def preview_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "preview"
    shutil.copytree(PROOF_ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root / "dinkster-pack.toml"


@pytest.mark.parametrize(
    "event_name", [PROOF_EVENT, "video_preview.some_event", "video-preview.nested.some_event"]
)
def test_frontend_consumer_preserves_declared_event_name(
    preview_manifest: Path, event_name: str
) -> None:
    preview_manifest.write_text(preview_manifest.read_text().replace(PROOF_EVENT, event_name))
    manifest = load_manifest(preview_manifest)
    module = resolve_frontend_modules(manifest)[0]
    assert manifest.extension.events[0].name == event_name
    assert module.contributions[0].event == event_name
    assert module.to_wire(PROOF_PACK)["contributions"] == [
        {
            "id": "dinkster-video-preview.preview.initialized",
            "kind": "eventConsumer",
            "event": event_name,
        },
        {"id": "dinkster-video-preview.preview.status", "kind": "hostUi"},
    ]
    undeclared = replace(manifest, extension=replace(manifest.extension, events=()))
    with pytest.raises(ValueError, match="event declared by its pack"):
        resolve_frontend_modules(undeclared)


@pytest.mark.parametrize(
    "event_name", [None, 1, "", "example", "example..event", "Example.event", "example.1event"]
)
def test_frontend_consumer_and_pack_event_reject_the_same_invalid_names(event_name: str) -> None:
    from dinkster_protocol import PackEvent
    from dinkster_protocol.frontend_modules import FrontendContribution

    with pytest.raises(ValueError, match="pack event name"):
        PackEvent(event_name)
    with pytest.raises(ValueError, match="pack event name"):
        FrontendContribution("example.consumer", "eventConsumer", event_name)


def test_event_names_do_not_relax_frontend_contribution_ids() -> None:
    from dinkster_protocol.frontend_modules import FrontendContribution

    with pytest.raises(ValueError, match="frontend id"):
        FrontendContribution("example.some_consumer", "eventConsumer", "example.some_event")
    with pytest.raises(ValueError, match="only eventConsumer"):
        FrontendContribution("example.consumer", "hostUi", "example.some_event")


def test_catalog_snapshot_and_module_do_not_activate_worker(preview_manifest: Path) -> None:
    env = {"PYTHONPATH": str(preview_manifest.parent / "src")}
    report = diagnose(preview_manifest, environment=env)
    assert report.ok, report
    manifest = load_manifest(preview_manifest)
    catalog = read_catalog(manifest)
    assert catalog is not None
    assert (
        tuple(route for _, item in catalog.extension_contributions for route in item.routes)
        == manifest.extension.routes
    )
    prior_imports = set(sys.modules)

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(preview_manifest, require_catalog=True, env=env))
            worker = composer._records[PROOF_PACK].worker
            assert worker.cold and not worker.alive
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                pack_route_dispatch=composer.call_pack_route,
                frontend_module_read=composer.read_frontend_module,
            )
            async with TestClient(TestServer(app)) as client:
                nodes = await (await client.get("/api/nodes")).json()
                raw = await (await client.get("/api/extensions/snapshot")).read()
                assert (
                    nodes["extensionSnapshotDigest"] == "sha256:" + hashlib.sha256(raw).hexdigest()
                )
                extension = json.loads(raw)["extensions"][0]
                assert extension["events"][0]["name"] == PROOF_EVENT
                module = extension["frontend"][0]
                response = await client.get(module["moduleUrl"])
                assert response.status == 200
                code = await response.read()
                assert module["moduleDigest"] == "sha256:" + hashlib.sha256(code).hexdigest()
                assert "immutable" in response.headers["Cache-Control"]
                assert response.headers["X-Content-Type-Options"] == "nosniff"
                assert (await client.head(module["moduleUrl"])).status == 200
                assert worker.cold and not worker.alive
                assert not (set(sys.modules) - prior_imports) & {
                    "torch",
                    "comfy",
                    "dinkster_video_preview",
                }
                assert (
                    await client.get(PROOF_ROUTE, headers={"If-Match": "sha256:" + "0" * 64})
                ).status == 412
                assert worker.cold and not worker.alive
                response = await client.get(
                    PROOF_ROUTE, headers={"If-Match": nodes["extensionSnapshotDigest"]}
                )
                assert response.status == 200, await response.text()
                assert (
                    response.headers["X-Dinkster-Extension-Snapshot"]
                    == nodes["extensionSnapshotDigest"]
                )
                assert await response.json() == {
                    "defaultFps": 24.0,
                    "maxFrames": 120,
                    "maxWidth": 512,
                }
                assert worker.alive
                assert worker_declarations(worker) == worker_declarations(catalog)
                assert (await client.post(PROOF_ROUTE, json={})).status == 405
                assert (await client.get(PROOF_ROUTE + "?path=/etc/passwd")).status == 400
                assert (await client.get(PROOF_ROUTE + "-missing")).status == 404
                (preview_manifest.parent / "src/dinkster_video_preview/preview.js").write_text(
                    "changed"
                )
                assert (await client.get(module["moduleUrl"])).status == 404
                await composer.remove_pack(PROOF_PACK)
                assert (await client.get(PROOF_ROUTE)).status == 404
                assert (await client.get(module["moduleUrl"])).status == 404
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_module_symlinks_and_digest_changes_are_rejected(
    preview_manifest: Path, tmp_path: Path
) -> None:
    manifest = load_manifest(preview_manifest)
    module = resolve_frontend_modules(manifest)[0]
    path = preview_manifest.parent / module.module
    outside = tmp_path / path.name
    outside.write_text("export const secret = 1;")
    path.unlink()
    if sys.platform == "win32":
        path.parent.rename(tmp_path / "original-module")
        subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(path.parent), str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        path.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="escapes"):
            read_module(manifest, module)
    finally:
        if sys.platform == "win32":
            path.parent.rmdir()
        else:
            path.unlink()


def test_http_route_capabilities_json_validation_and_generation_retraction() -> None:
    schema = JsonObjectSchema((JsonField("count", "integer"),))
    route = PackRoute("count", "POST", "example:count", schema, schema)
    snapshot = ExtensionSnapshot((ActiveExtension("example", "1.0.0", "digest", routes=(route,)),))
    calls: list[object] = []

    async def dispatch(pack, declaration, data, digest):
        calls.append((pack, declaration, data))
        return data

    async def scenario() -> None:
        nonlocal snapshot
        app = web.Application()
        capabilities = install_pack_surfaces(app, lambda: snapshot, dispatch, None)
        auth = StaticBearerAuthenticator(
            {
                "reader": Principal("reader", grants={"local": frozenset({"jobs:read"})}),
                "writer": Principal("writer", grants={"local": frozenset({"jobs:submit"})}),
            }
        )
        install_auth(app, auth, route_capabilities=capabilities)
        path = "/api/extensions/example/routes/count"
        async with TestClient(TestServer(app)) as client:
            assert (await client.post(path, json={"count": 1})).status == 401
            denied = await client.post(
                path, json={"count": 1}, headers={"Authorization": "Bearer reader"}
            )
            assert denied.status == 403
            assert (await denied.json())["capability"] == "jobs:submit"
            client.session.headers["Authorization"] = "Bearer writer"
            for data in ({}, {"count": True}, {"count": 1, "extra": 2}):
                assert (await client.post(path, json=data)).status == 400
            assert (
                await client.post(path, data="{}", headers={"Content-Type": "text/plain"})
            ).status == 415
            assert (
                await client.post(
                    path, data="{" + "x" * 65536, headers={"Content-Type": "application/json"}
                )
            ).status == 413
            assert calls == []
            response = await client.post(path, json={"count": 4})
            assert response.status == 200
            assert await response.json() == {"count": 4}
            assert calls == [("example", route, {"count": 4})]
            snapshot = ExtensionSnapshot()
            assert (await client.post(path, json={"count": 4})).status == 404
            assert len(calls) == 1

    asyncio.run(scenario())


def test_event_contract_requires_finite_bounded_json() -> None:
    schema = JsonObjectSchema((JsonField("value", "number"),))
    for value in (float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError):
            schema.validate({"value": value})
    text = JsonObjectSchema((JsonField("value", "string"),))
    with pytest.raises(ValueError, match="64 KiB"):
        text.validate({"value": "x" * 65536})
    assert schema.validate({"value": 1.5}) == {"value": 1.5}


@pytest.mark.parametrize("field_type", ["integer", "number"])
@pytest.mark.parametrize("value", SAFE_JSON_INTS + UNSAFE_JSON_INTS)
def test_schema_and_event_reporter_enforce_safe_integer_bounds(field_type: str, value: int) -> None:
    from dinkster_api.v1 import PackEvent, report_pack_event
    from dinkster_schema import use_reporter

    schema = JsonObjectSchema((JsonField("value", field_type),))
    event = PackEvent(PROOF_EVENT, schema)
    payload = {"value": value}
    received: list[object] = []
    with use_reporter(lambda name, data, blob: received.append((name, data, blob))):
        if value in SAFE_JSON_INTS:
            assert schema.validate(payload) == payload
            report_pack_event(event, payload)
            assert received == [(PROOF_EVENT, payload, None)]
        else:
            with pytest.raises(ValueError, match="JSON safe integer"):
                schema.validate(payload)
            with pytest.raises(ValueError, match="JSON safe integer"):
                report_pack_event(event, payload)
            assert received == []


@pytest.mark.parametrize("value", [-1e100, -float(2**53), float(2**53), 1e100, 1.25])
def test_number_schema_preserves_finite_floats(value: float) -> None:
    schema = JsonObjectSchema((JsonField("value", "number"),))
    result = schema.validate({"value": value})
    assert result == {"value": value}
    assert type(result["value"]) is float


@pytest.mark.parametrize("field_type", ["integer", "number"])
def test_http_route_request_and_response_enforce_safe_integer_bounds(field_type: str) -> None:
    schema = JsonObjectSchema((JsonField("value", field_type),))
    route = PackRoute("number", "POST", "example:number", schema, schema)
    snapshot = ExtensionSnapshot((ActiveExtension("example", "1", "digest", routes=(route,)),))
    response_value = 0
    calls: list[object] = []

    async def dispatch(pack, declaration, data, digest) -> dict[str, object]:
        calls.append(data)
        return {"value": response_value}

    async def scenario() -> None:
        nonlocal response_value
        app = web.Application()
        install_pack_surfaces(app, lambda: snapshot, dispatch, None)
        path = "/api/extensions/example/routes/number"
        async with TestClient(TestServer(app)) as client:
            for value in SAFE_JSON_INTS:
                response_value = value
                response = await client.post(path, json={"value": value})
                assert response.status == 200
                assert await response.json() == {"value": value}
                assert calls[-1] == {"value": value}
            for value in UNSAFE_JSON_INTS:
                count = len(calls)
                response = await client.post(path, json={"value": value})
                assert response.status == 400
                assert await response.json() == {"error": "pack-route-invalid-request"}
                assert len(calls) == count
                response_value = value
                response = await client.post(path, json={"value": 0})
                assert response.status == 502
                assert await response.json() == {"error": "pack-route-failed"}
                assert len(calls) == count + 1

    asyncio.run(scenario())


def test_same_pack_catalog_identity_and_stale_module_rejection(preview_manifest: Path) -> None:
    env = {"PYTHONPATH": str(preview_manifest.parent / "src")}
    assert diagnose(preview_manifest, environment=env).ok
    other = preview_manifest.with_name("native-pack.toml")
    shutil.copyfile(preview_manifest, other)
    # Creating another source file invalidates the first artifact's source digest.
    assert diagnose(preview_manifest, environment=env).ok
    assert diagnose(other, environment=env).ok
    first, second = (
        read_catalog(load_manifest(preview_manifest)),
        read_catalog(load_manifest(other)),
    )
    assert first is not None and second is not None and first.source != second.source
    manifest = load_manifest(preview_manifest)
    module = resolve_frontend_modules(manifest)[0]
    assert module != replace(module, module_digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="snapshot"):
        read_module(manifest, replace(module, module_digest="sha256:" + "0" * 64))


@pytest.mark.parametrize("field_type", ["integer", "number"])
def test_typed_event_uses_pinned_generation_and_rejects_invalid_payloads(field_type: str) -> None:
    from dinkster_engine import EngineEvent, ExecutionSelection
    from dinkster_protocol import InvocationEvent, PackEvent

    from tests.test_engine_dispatch import RecordingWorker, _engine, _graph, _string

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        class ReportingWorker(RecordingWorker):
            async def invoke(self, invocation, on_event=None):
                entered.set()
                await release.wait()
                assert on_event is not None
                on_event(InvocationEvent(PROOF_EVENT, {"count": 4}))
                for value in SAFE_JSON_INTS + UNSAFE_JSON_INTS:
                    on_event(InvocationEvent(PROOF_EVENT, {"count": value}))
                on_event(InvocationEvent(PROOF_EVENT, {"count": True}))
                on_event(InvocationEvent(PROOF_EVENT, {"count": 4}, blob=b"no binary"))
                on_event(InvocationEvent("foreign.event", {"count": 4}))
                on_event(InvocationEvent("legacy.notice", {"text": "unchanged"}))
                return await super().invoke(invocation, on_event)

        async def plan(*args):
            return ExecutionSelection("native", "preview-v1", worker="local", pack=PROOF_PACK)

        events: list[EngineEvent] = []
        worker = ReportingWorker(lambda n: _string("done"))
        engine = _engine(worker, events=events, plan_execution=plan)
        schema = JsonObjectSchema((JsonField("count", field_type),))
        snapshot = ExtensionSnapshot(
            (
                ActiveExtension(
                    PROOF_PACK, "1", "digest", events=(PackEvent(PROOF_EVENT, schema),)
                ),
                ActiveExtension(
                    "foreign", "1", "digest", events=(PackEvent("foreign.event", schema),)
                ),
            )
        )
        runtime = replace(engine.pin_execution(), extension_snapshot=snapshot)
        task = asyncio.create_task(engine.run(_graph(), ["p"], execution=runtime))
        await entered.wait()
        assert engine.extension_snapshot == ExtensionSnapshot()
        release.set()
        await task
        emitted = [event for event in events if event.kind == "node_event"]
        assert [event.detail["name"] for event in emitted] == [PROOF_EVENT] * 3 + ["legacy.notice"]
        assert [event.detail["data"] for event in emitted[:3]] == [
            {"count": value} for value in (4, *SAFE_JSON_INTS)
        ]
        assert emitted[0].detail == {
            "name": PROOF_EVENT,
            "data": {"count": 4},
            "pack": PROOF_PACK,
            "worker": "local",
            "executionArm": "native",
            "schemaVersion": 1,
            "extensionSnapshotDigest": runtime.extension_snapshot_digest,
        }
        assert "schemaVersion" not in emitted[-1].detail

    asyncio.run(scenario())


@pytest.mark.parametrize("close", [False, True])
def test_route_rpc_cancellation_and_worker_death_clear_pending(
    monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    from dinkster_workers.session import WorkerDied

    from tests.test_isolated import graph_compile_session

    session = graph_compile_session()
    sent: list[dict[str, object]] = []

    async def send(header, blobs):
        sent.append(header)

    monkeypatch.setattr(session, "send", send)

    async def scenario() -> None:
        route = PackRoute("policy", "GET", "pack:policy")
        task = asyncio.create_task(session.call_pack_route(route, {}))
        await asyncio.sleep(0)
        assert sent == [
            {"type": "packRoute", "requestId": "route-0", "route": route.to_wire(), "data": {}}
        ]
        if close:
            await session.close()
            with pytest.raises(WorkerDied):
                await task
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert sent[-1] == {"type": "cancelPackRoute", "requestId": "route-0"}
        assert session._route_pending == {}

    asyncio.run(scenario())
