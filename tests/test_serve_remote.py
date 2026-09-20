"""Remote workers on the composed serving surface: add_remote folds a
dinkster_workers.service daemon's announced node types into a ServingComposer
exactly like a pack - attribution through node_packs, provenance through
PackInfo.source, execution routed over the wire with events intact - and
every misconfiguration (schema mismatch, allowlist miss, reserved root without
trust, dead daemon) is refused ATOMICALLY: registries untouched, the
connection closed, the daemon free for the next client."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetVault, digest_bytes
from dinkster_engine import EngineEvent, ExecutionError, ExecutionSelection
from dinkster_graph import (
    PORTS_NODE_ID,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    graph_to_wire,
)
from dinkster_protocol import AttentionRouteToken
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComfyAliasConfidence,
    ComfyAliasRecord,
    ComfyAliasRegistry,
    ComfyAliasSource,
    ComfyAliasSourceSchema,
    ComfyGroupEdge,
    ComfyGroupNode,
    ComfyGroupPattern,
    ComfyGroupRecord,
    ComfyGroupRegistry,
    ComfyGroupSource,
    ComfyGroupSourceSchema,
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    comfy_alias_registry_to_wire,
    comfy_group_registry_to_wire,
)
from dinkster_server import STATE_KEY, PackInfo, create_app
from dinkster_values import CORE_STRING, list_children, value_resource_provenance_refs
from dinkster_workers import RemoteWorker, TransportError
from dinkster_workers.service import READY_LINE_PREFIX
from test_choices import write_choice_manifest
from test_compose_dispatch import _write_generation_provider_packs, _write_vision_provider_packs
from test_isolated import DEV_MANIFEST, core_registry, write_iso_manifest
from test_remote import TLS_ARGS, TLS_CERT, TOKEN, start_service, stop_service

from dinkster.compose import (
    CompositionError,
    PackSpec,
    ServingComposer,
    _validated_remote_body_arms,
)
from dinkster.remotes import RemoteSpec
from dinkster.serve import merge_remote_budgets

TESTS_DIR = Path(__file__).parent
MEDIA_IO_MANIFEST = TESTS_DIR.parent / "packages" / "dinkster-nodes-media-io" / "dinkster-pack.toml"


def remote_spec(host: str, port: int, tmp_path: Path, **overrides: object) -> RemoteSpec:
    """A spec pointing at a daemon start_service launched (its token file
    lives at tmp_path/remote-token.txt)."""
    fields: dict[str, object] = {
        "name": "box1",
        "host": host,
        "port": port,
        "token_file": tmp_path / "remote-token.txt",
    }
    fields.update(overrides)
    return RemoteSpec(**fields)  # type: ignore[arg-type]


async def daemon_accepts_a_client(host: str, port: int) -> None:
    """Prove the daemon's single conversation slot is free: a fresh client
    must complete the handshake (a leaked composer connection would hold
    the slot and time this out)."""
    probe = RemoteWorker(host, port, TOKEN, core_registry(), name="probe", connect_timeout=10.0)
    await probe.start()
    await probe.close()


def composition_snapshot(composer: ServingComposer) -> tuple[object, ...]:
    composition = composer.composition
    return (
        dict(composition.schemas),
        dict(composition.packs),
        dict(composition.node_packs),
        dict(composition.choices),
        len(composition._isolated),
        dict(composer._remotes),
        dict(composer._owners),
        dict(composer._seen_names),
        dict(composer._choice_owners),
        dict(composer._compat_skip_owners),
    )


async def wait_for_job(client: TestClient, client_id: str, job_id: str) -> dict[str, object]:
    async with asyncio.timeout(10):
        while True:
            response = await client.get(f"/api/jobs/{client_id}/{job_id}")
            body = await response.json()
            if body["state"] in ("completed", "failed", "cancelled"):
                return body
            await asyncio.sleep(0.01)


def write_composed_iso_manifest(root: Path, name: str) -> Path:
    root.mkdir()
    manifest = write_iso_manifest(root, name=name)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace(f'name = "{name}"', f'name = "{name}"\nnamespaces = ["iso"]'),
        encoding="utf-8",
    )
    return manifest


def test_add_remote_composes_surface_and_executes_remotely(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        try:
            composer = ServingComposer()
            try:
                delta = await composer.add_remote(remote_spec(host, port, tmp_path))
                composition = composer.composition
                assert delta.pack == "box1"
                assert "iso.chatty" in composition.schemas
                assert composition.node_packs["iso.chatty"] == "box1"
                assert composition.packs["box1"].source == f"remote:{host}:{port}"
                assert composition.packs["box1"].version == ""

                events: list[EngineEvent] = []
                engine = composition.make_engine(on_event=events.append)
                graph = Graph(nodes={"c": GraphNode("iso.chatty", {"value": "hello"})})
                result = await engine.run(
                    graph,
                    ["c"],
                    run_id="hinted",
                    execution=composer.place_execution(
                        engine.pin_execution(),
                        {"c": "box1"},
                    ),
                )
                assert result.outputs["c"]["value"].resolve() == "HELLO"
                progress = [
                    event
                    for event in events
                    if event.kind == "node_event" and event.detail.get("name") == "progress"
                ]
                assert progress, "remote report_progress must reach the engine's listener"
                assert all(event.detail["worker"] == "box1" for event in progress)

                unhinted = await engine.run(graph, ["c"], run_id="unhinted")
                assert unhinted.cached == ("c",)
                keys = {
                    event.run_id: event.detail["cache_key"]
                    for event in events
                    if event.kind in ("node_finished", "node_cached") and event.node_id == "c"
                }
                assert keys["hinted"] == keys["unhinted"]
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_add_remote_composes_full_media_pack_with_lazy_capture_choices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        monkeypatch.delenv("DINKSTER_AUDIO_CAPTURE_PROVIDER", raising=False)
        monkeypatch.delenv("DINKSTER_VIDEO_CAPTURE_PROVIDER", raising=False)
        proc, host, port = await start_service(MEDIA_IO_MANIFEST, tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path, trust_reserved=True))
                composition = composer.composition
                choice_ids = {
                    "dinkster.devices.audio_inputs",
                    "dinkster.devices.video_inputs",
                }
                assert "dinkster.record_audio" in composition.schemas
                assert "dinkster.webcam_capture" in composition.schemas
                assert set(composition.lazy_choices) == choice_ids
                assert {
                    choice_id: composer._choice_owners[choice_id] for choice_id in choice_ids
                } == dict.fromkeys(choice_ids, "box1")
                for choice_id in choice_ids:
                    assert await composition.lazy_choices[choice_id]() == ()

                engine = composition.make_engine(on_event=lambda event: None)
                graph = Graph(
                    nodes={
                        "empty": GraphNode(
                            "dinkster.empty_audio",
                            {"duration": 0.001, "sample_rate": 8_000, "channels": 1},
                        )
                    }
                )
                result = await engine.run(
                    graph,
                    ["empty"],
                    execution=composer.place_execution(engine.pin_execution(), {"empty": "box1"}),
                )
                assert result.executed == ("empty",)
                assert result.outputs["empty"]["audio"].type_id == "comfy.AUDIO"
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_add_remote_composes_eager_choices_from_service(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_choice_manifest(tmp_path / "pack")
        proc, host, port = await start_service(manifest, tmp_path)
        try:
            composer = ServingComposer()
            try:
                delta = await composer.add_remote(remote_spec(host, port, tmp_path))
                assert delta.choices == {
                    "cp.empty": (),
                    "cp.samplers": ("euler", "ddim", "heun"),
                }
                assert not {"cp.empty", "cp.samplers"} & composer.composition.lazy_choices.keys()
                assert composer._choice_owners["cp.empty"] == "box1"
                assert composer._choice_owners["cp.samplers"] == "box1"
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_add_remote_over_tls_composes_and_executes(tmp_path: Path) -> None:
    """tls_ca_file in the spec reaches the connection: a TLS daemon is
    composed and executes exactly like a plaintext one."""

    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path, *TLS_ARGS)
        try:
            composer = ServingComposer()
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path, tls_ca_file=TLS_CERT))
                composition = composer.composition
                assert "iso.chatty" in composition.schemas
                engine = composition.make_engine(on_event=lambda event: None)
                graph = Graph(nodes={"c": GraphNode("iso.chatty", {"value": "tls"})})
                result = await engine.run(graph, ["c"])
                assert result.outputs["c"]["value"].resolve() == "TLS"
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_add_remote_refuses_unreadable_tls_ca_file_before_dialing(tmp_path: Path) -> None:
    """A misconfigured tls_ca_file fails with the file named, before any
    connection attempt: the endpoint here is unreachable, so a dial would
    produce a different error."""

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            token_file = tmp_path / "remote-token.txt"
            token_file.write_text(TOKEN, encoding="utf-8")
            spec = remote_spec("127.0.0.1", 1, tmp_path, tls_ca_file=tmp_path / "absent-ca.pem")
            with pytest.raises(CompositionError, match="cannot read TLS CA file"):
                await composer.add_remote(spec)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_submission_hint_splits_workers_and_separates_worker_cache_identity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        local_manifest = write_composed_iso_manifest(tmp_path / "local", "local-iso")
        remote_root = tmp_path / "remote"
        remote_manifest = write_composed_iso_manifest(remote_root, "remote-iso")
        proc, host, port = await start_service(remote_manifest, remote_root)
        try:
            composer = ServingComposer(worker_env={"PYTHONPATH": str(TESTS_DIR)})
            client: TestClient | None = None
            try:
                await composer.add_pack(local_manifest)
                spec = remote_spec(host, port, remote_root)
                await composer.add_remote(spec)
                composition = composer.composition
                app = create_app(
                    composition.make_engine,
                    composition.schemas,
                    workers=lambda: composer.workers((spec,)),
                    place_execution=composer.place_execution,
                )
                client = TestClient(TestServer(app))
                await client.start_server()

                workers = await (await client.get("/api/workers")).json()
                by_name = {entry["name"]: entry for entry in workers["workers"]}
                assert by_name["local"]["status"] == "connected"
                assert by_name["box1"] == {
                    "name": "box1",
                    "status": "connected",
                    "routedNodeTypes": sorted(composer._remotes["box1"].node_types),
                    "deviceQualifiers": ["@box1"],
                }

                split = Graph(
                    nodes={
                        "light": GraphNode("iso.chatty", {"value": "light"}),
                        "heavy": GraphNode("iso.chatty", {"value": "heavy"}),
                    }
                )
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "split",
                        "jobId": "one",
                        "graph": graph_to_wire(split),
                        "targets": ["light", "heavy"],
                        "placement": {"heavy": "box1"},
                    },
                )
                assert response.status == 202
                submitted = await response.json()
                assert (await wait_for_job(client, "split", "one"))["state"] == "completed"
                job = app[STATE_KEY].queue.get("split", "one")
                assert job is not None and job.result is not None
                assert job.result.outputs["light"]["value"].resolve() == "LIGHT"
                assert job.result.outputs["heavy"]["value"].resolve() == "HEAVY"
                assert job.node_receipts["light"]["worker"] == "local"
                assert job.node_receipts["heavy"]["worker"] == "box1"
                replay = await (
                    await client.get(f"/api/jobs/by-ref/{submitted['jobRef']}/events?after=0")
                ).json()
                finished = {
                    event["nodeId"]: event["detail"]["worker"]
                    for event in replay["events"]
                    if event["type"] == "node_finished"
                }
                assert finished == {"light": "local", "heavy": "box1"}
                remote_reports = [
                    event
                    for event in replay["events"]
                    if event["type"] == "node_event" and event["nodeId"] == "heavy"
                ]
                assert remote_reports
                assert {event["worker"] for event in remote_reports} == {"box1"}
                assert {event["executionArm"] for event in remote_reports} == {"native"}
                assert any(event["event"] == "progress" for event in remote_reports)
                preview = next(event for event in remote_reports if event["event"] == "preview")
                assert preview["blobOmitted"] is True

                cache_graph = Graph(
                    nodes={"cached": GraphNode("iso.chatty", {"value": "same-work"})}
                )
                for client_id, placement in (
                    ("cache-local", None),
                    ("cache-remote", {"cached": "box1"}),
                ):
                    body: dict[str, object] = {
                        "clientId": client_id,
                        "jobId": "one",
                        "graph": graph_to_wire(cache_graph),
                        "targets": ["cached"],
                    }
                    if placement is not None:
                        body["placement"] = placement
                    assert (await client.post("/api/jobs", json=body)).status == 202
                    assert (await wait_for_job(client, client_id, "one"))["state"] == "completed"
                local_job = app[STATE_KEY].queue.get("cache-local", "one")
                remote_job = app[STATE_KEY].queue.get("cache-remote", "one")
                assert local_job is not None and remote_job is not None
                assert local_job.node_receipts["cached"]["disposition"] == "executed"
                assert remote_job.node_receipts["cached"] == {
                    "nodeId": "cached",
                    "disposition": "executed",
                    "executionArm": "native",
                    "worker": "box1",
                    "pack": "remote-iso",
                }
                assert app[STATE_KEY].run_cache_key(local_job.run_id, "cached") != (
                    app[STATE_KEY].run_cache_key(remote_job.run_id, "cached")
                )
            finally:
                if client is not None:
                    await client.close()
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_vision_provider_preflights_with_consent_and_reports_actual_execution(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        model_bytes = b"provider-model"
        model_digest = digest_bytes(model_bytes)
        source_requests = 0

        async def serve_model(_request: web.Request) -> web.Response:
            nonlocal source_requests
            source_requests += 1
            return web.Response(body=model_bytes)

        source_app = web.Application()
        source_app.router.add_get("/model.bin", serve_model)
        source = TestServer(source_app)
        await source.start_server()
        owner_manifest, provider_manifest = _write_vision_provider_packs(tmp_path / "packs")
        provider_manifest.write_text(
            provider_manifest.read_text(encoding="utf-8").replace(
                'file = "assets/model.bin"',
                f'urls = ["{source.make_url("/model.bin")}"]',
            ),
            encoding="utf-8",
        )
        daemon_root = tmp_path / "daemon"
        daemon_root.mkdir()
        daemon_vault = daemon_root / "vault"
        proc, host, port = await start_service(
            provider_manifest,
            daemon_root,
            "--asset-vault",
            str(daemon_vault),
            pythonpath=(provider_manifest.parent,),
        )
        try:
            composer = ServingComposer(worker_env={"PYTHONPATH": str(owner_manifest.parent)})
            client: TestClient | None = None
            try:
                await composer.add_pack(owner_manifest)
                spec = remote_spec(host, port, daemon_root)
                await composer.add_remote(spec)
                remote = composer._remotes["box1"].worker
                assert remote.vision_providers is not None
                assert remote.vision_providers[0].artifacts == ("model",)
                assert remote.generation_providers == ()

                composition = composer.composition
                app = create_app(
                    composition.make_engine,
                    composition.schemas,
                    choices=composition.choices,
                    workers=lambda: composer.workers((spec,)),
                    place_execution=composer.place_execution,
                )
                client = TestClient(TestServer(app))
                await client.start_server()
                graph = Graph(nodes={"n": GraphNode("vision.process", {"value": "remote"})})
                body = {
                    "clientId": "provider",
                    "jobId": "first",
                    "graph": graph_to_wire(graph),
                    "targets": ["n"],
                    "placement": {"n": "box1"},
                }

                refused = await client.post("/api/jobs", json=body)
                refused_body = await refused.json()
                assert refused.status == 409, refused_body
                assert refused_body == {
                    "error": "assets-missing",
                    "assets": [
                        {
                            "digest": model_digest,
                            "name": "Test model",
                            "status": "missing",
                            "sources": [str(source.make_url("/model.bin"))],
                            "fetchable": True,
                        }
                    ],
                }
                assert source_requests == 0
                assert not AssetVault(daemon_vault).has(model_digest)
                assert app[STATE_KEY].queue.jobs() == []

                body["acquireAssets"] = [model_digest]
                accepted = await client.post("/api/jobs", json=body)
                assert accepted.status == 202
                status = await wait_for_job(client, "provider", "first")
                assert status["state"] == "completed"
                job = app[STATE_KEY].queue.get("provider", "first")
                assert job is not None and job.result is not None
                assert job.result.outputs["n"]["value"].resolve() == "provider-model:remote"
                assert job.node_receipts["n"] == {
                    "nodeId": "n",
                    "disposition": "executed",
                    "executionArm": "native",
                    "worker": "box1",
                    "provider": "depth-provider",
                    "pack": "depth-provider",
                }
                assert source_requests == 1
                assert AssetVault(daemon_vault).has(model_digest)

                app[STATE_KEY].queue.pause()
                body["jobId"] = "asset-disappeared"
                body["graph"] = graph_to_wire(
                    Graph(nodes={"n": GraphNode("vision.process", {"value": "remote-again"})})
                )
                queued = await client.post("/api/jobs", json=body)
                assert queued.status == 202
                for path in daemon_vault.rglob("*"):
                    if path.is_file():
                        path.unlink()
                app[STATE_KEY].queue.resume()
                disappeared = await wait_for_job(client, "provider", "asset-disappeared")
                assert disappeared["state"] == "failed"
                error = disappeared["error"]
                assert isinstance(error, dict)
                assert error["nodeId"] == "n"
                message = error["message"]
                assert isinstance(message, str)
                assert "provider assets disappeared after job admission" in message
                assert model_digest in message
                assert "resubmit the job" in message
                assert source_requests == 1
            finally:
                if client is not None:
                    await client.close()
                await composer.close()
        finally:
            await stop_service(proc)
            await source.close()

    asyncio.run(scenario())


def test_remote_generation_provider_uses_hidden_routing_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_manifest, native_manifest, provider_manifest = _write_generation_provider_packs(
            tmp_path / "packs"
        )
        daemon_root = tmp_path / "daemon"
        daemon_root.mkdir()
        proc, host, port = await start_service(
            provider_manifest,
            daemon_root,
            pythonpath=(provider_manifest.parent,),
        )
        try:
            composer = ServingComposer(
                worker_env={
                    "PYTHONPATH": os.pathsep.join(
                        (str(owner_manifest.parent), str(native_manifest.parent))
                    )
                }
            )
            try:
                await composer.add_pack(owner_manifest)
                await composer.add_pack(native_manifest)
                delta = await composer.add_remote(remote_spec(host, port, daemon_root))
                remote = composer._remotes["box1"].worker
                assert remote.generation_providers is not None
                assert remote.generation_providers[0].node == "fixture.generate"
                assert delta.derived_choices == {
                    "fixture.generation.providers": ("generation-external",)
                }
                provider_input = composer.composition.schemas["fixture.generate"].input("provider")
                assert provider_input is not None
                assert provider_input.hidden
                assert not provider_input.advanced

                events: list[EngineEvent] = []
                engine = composer.composition.make_engine(events.append)
                graph = Graph(
                    nodes={
                        "generate": GraphNode(
                            "fixture.generate",
                            {
                                "provider": "generation-external",
                                "value": "remote prompt",
                            },
                        )
                    }
                )
                result = await engine.run(graph, ["generate"])
                assert result.outputs["generate"]["value"].resolve() == "external:remote prompt"
                finished = next(
                    event
                    for event in events
                    if event.kind == "node_finished" and event.node_id == "generate"
                )
                assert finished.detail["worker"] == "box1"
                assert finished.detail["provider"] == "generation-external"
                assert finished.detail["pack"] == "generation-external"
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_hint_overrides_policy_requiring_an_unavailable_native_arm(tmp_path: Path) -> None:
    class RequiresNativeArm:
        calls = 0

        async def select(
            self,
            node_type: str,
            inputs: object,
            arms: tuple[str, object],
            *,
            run_id: str | None = None,
            extension_behavior_hash: str | None = None,
            attention_routes: object,
        ) -> ExecutionSelection:
            del node_type, inputs, run_id, extension_behavior_hash, attention_routes
            self.calls += 1
            return ExecutionSelection(
                target=f"{arms[0]}@native",
                cache_tag="policy-native",
            )

    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        try:
            policy = RequiresNativeArm()
            composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path))
                engine = composer.composition.make_engine(lambda event: None)
                graph = Graph(nodes={"c": GraphNode("iso.chatty", {"value": "hinted"})})
                result = await engine.run(
                    graph,
                    ["c"],
                    execution=composer.place_execution(
                        engine.pin_execution(),
                        {"c": "box1"},
                    ),
                )
                assert result.outputs["c"]["value"].resolve() == "HINTED"
                # The policy is consulted (managed types derive their
                # execution identity from its selection), but its choice of
                # an arm this remote does not offer is overridden by the hint.
                assert policy.calls == 1
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_hint_dispatch_carries_policy_execution_identity(tmp_path: Path) -> None:
    """A single-arm remote hint still consults the policy, and an in-set
    selection's cache tag crosses the wire as the invocation's expected
    execution identity (the identity native loads verify against)."""

    class IdentityPolicy:
        calls = 0

        async def select(
            self,
            node_type: str,
            inputs: object,
            arms: tuple[str, object],
            *,
            run_id: str | None = None,
            extension_behavior_hash: str | None = None,
            attention_routes: object,
        ) -> ExecutionSelection:
            del node_type, inputs, run_id, extension_behavior_hash, attention_routes
            self.calls += 1
            return ExecutionSelection(
                target=arms[0],
                cache_tag="policy-identity",
            )

    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        try:
            policy = IdentityPolicy()
            composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path))
                engine = composer.composition.make_engine(lambda event: None)
                graph = Graph(nodes={"probe": GraphNode("iso.identity_echo", {})})
                result = await engine.run(
                    graph,
                    ["probe"],
                    execution=composer.place_execution(
                        engine.pin_execution(),
                        {"probe": "box1"},
                    ),
                )
                assert result.outputs["probe"]["identity"].resolve() == "policy-identity"
                assert policy.calls == 1
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_hint_does_not_share_cache_across_different_execution_identities(tmp_path: Path) -> None:
    async def scenario() -> None:
        local_manifest = write_composed_iso_manifest(tmp_path / "local", "local-iso")
        remote_root = tmp_path / "remote"
        remote_manifest = write_composed_iso_manifest(remote_root, "remote-iso")
        proc, host, port = await start_service(remote_manifest, remote_root)
        try:
            composer = ServingComposer(worker_env={"PYTHONPATH": str(TESTS_DIR)})
            try:
                await composer.add_pack(
                    PackSpec(
                        manifest=local_manifest,
                        packs={
                            "local-iso": PackInfo(
                                display_name="Local ISO",
                                artifact_digest="sha256:" + "a" * 64,
                            )
                        },
                    )
                )
                await composer.add_remote(remote_spec(host, port, remote_root))
                events: list[EngineEvent] = []
                engine = composer.composition.make_engine(events.append)
                graph = Graph(nodes={"work": GraphNode("iso.chatty", {"value": "same"})})

                local = await engine.run(graph, ["work"], run_id="local")
                remote = await engine.run(
                    graph,
                    ["work"],
                    run_id="remote",
                    execution=composer.place_execution(
                        engine.pin_execution(),
                        {"work": "box1"},
                    ),
                )
                assert local.executed == ("work",)
                assert remote.executed == ("work",)
                keys = {
                    event.run_id: event.detail["cache_key"]
                    for event in events
                    if event.kind == "node_finished" and event.node_id == "work"
                }
                assert keys["local"] != keys["remote"]
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_local_remote_name_is_reserved_before_connect(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            with pytest.raises(CompositionError, match="remote worker name 'local' is reserved"):
                await composer.add_remote(remote_spec("127.0.0.1", 1, tmp_path, name="local"))
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_region_hint_is_recursive_and_residency_conflict_is_loud(tmp_path: Path) -> None:
    async def scenario() -> None:
        local_manifest = write_composed_iso_manifest(tmp_path / "local", "local-iso")
        remote_root = tmp_path / "remote"
        remote_manifest = write_composed_iso_manifest(remote_root, "remote-iso")
        proc, host, port = await start_service(remote_manifest, remote_root)
        try:
            composer = ServingComposer(worker_env={"PYTHONPATH": str(TESTS_DIR)})
            try:
                await composer.add_pack(local_manifest)
                await composer.add_remote(remote_spec(host, port, remote_root))
                events: list[EngineEvent] = []
                engine = composer.composition.make_engine(events.append)
                string = TypeExpr.concrete(CORE_STRING)
                region = RegionNode(
                    kind="map",
                    body=Graph(
                        nodes={
                            "chat": GraphNode(
                                "iso.chatty",
                                {"value": Link(PORTS_NODE_ID, "item")},
                            )
                        }
                    ),
                    ports={"item": string},
                    inputs={"item": ["first", "second"]},
                    element_ports=("item",),
                    outputs={"values": RegionOutput(Link("chat", "value"))},
                )
                graph = Graph(nodes={"region": region})
                execution = composer.place_execution(engine.pin_execution(), {"region": "box1"})
                result = await engine.run(graph, ["region"], execution=execution)
                children = list_children(result.outputs["region"]["values"])
                assert children is not None
                assert [value.resolve() for value in children] == [
                    "FIRST",
                    "SECOND",
                ]
                body_workers = {
                    event.node_id: event.detail["worker"]
                    for event in events
                    if event.kind == "node_finished"
                    and isinstance(event.node_id, str)
                    and event.node_id.startswith("region[")
                }
                assert body_workers == {
                    "region[0]/chat": "box1",
                    "region[1]/chat": "box1",
                }

                conflict = Graph(
                    nodes={
                        "produce": GraphNode("iso.gpu_blob_out", {}),
                        "consume": GraphNode(
                            "iso.gpu_blob_identity",
                            {"blob": Link("produce", "blob")},
                        ),
                    }
                )
                conflict_execution = composer.place_execution(
                    engine.pin_execution(), {"consume": "box1"}
                )
                with pytest.raises(
                    ExecutionError,
                    match="resident state cannot cross workers",
                ):
                    await engine.run(conflict, ["consume"], execution=conflict_execution)
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_surface_survives_pack_reload(tmp_path: Path) -> None:
    """reload_pack rebuilds the registries from the live records AND the
    composed remotes: the remote's schemas, pack-table entry, ownership
    rows, and dispatch must all survive a pack reload intact."""

    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_pack(DEV_MANIFEST)
                await composer.add_remote(remote_spec(host, port, tmp_path))
                await composer.reload_pack("dinkster-nodes-dev")
                composition = composer.composition
                assert "iso.chatty" in composition.schemas
                assert composition.node_packs["iso.chatty"] == "box1"
                assert composition.packs["box1"].source == f"remote:{host}:{port}"
                assert composer._owners["iso.chatty"] == "box1"
                assert composer._seen_names.get("box1") == "box1"
                assert "iso.chatty" in composition.execution_arms

                engine = composition.make_engine(on_event=lambda event: None)
                graph = Graph(nodes={"c": GraphNode("iso.chatty", {"value": "post-reload"})})
                result = await engine.run(graph, ["c"])
                assert result.outputs["c"]["value"].resolve() == "POST-RELOAD"
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_matching_node_types_compose_as_placement_targets(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_pack(DEV_MANIFEST)
                before_schemas = dict(composer.composition.schemas)
                delta = await composer.add_remote(remote_spec(host, port, tmp_path))
                assert delta.schemas == {}
                assert composer.composition.schemas == before_schemas
                assert composer.composition.node_packs["dev.gallery.widgets"] == (
                    "dinkster-nodes-dev"
                )
                assert [arm.name for arm in composer._topology["dev.gallery.widgets"]] == [
                    "dinkster-nodes-dev",
                    "box1",
                ]
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_resident_producer_and_consumer_on_the_same_remote(tmp_path: Path) -> None:
    """The daemon's worker host stamps resident provenance with ITS local
    pack name ('respack'), which means nothing in this composition's arm
    namespace - the engine requalifies the session's outputs to the remote
    arm, so a same-remote consumer passes producer-arm affinity and the
    provenance carries the engine-side arm name."""

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "respack"\n\n[pack.entry]\n'
            'nodes = "respack_nodes:NODES"\ntypes = "respack_nodes:register_types"\n'
        )
        proc, host, port = await start_service(manifest, tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path))
                engine = composer.composition.make_engine(lambda event: None)
                graph = Graph(
                    nodes={
                        "produce": GraphNode("res.load", {"token": "secret"}),
                        "consume": GraphNode(
                            "res.use",
                            {
                                "heavy": Link("produce", "heavy"),
                                "oid": Link("produce", "oid"),
                            },
                        ),
                    }
                )
                result = await engine.run(
                    graph,
                    ["produce", "consume"],
                    execution=composer.place_execution(
                        engine.pin_execution(),
                        {"produce": "box1", "consume": "box1"},
                    ),
                )
                assert result.outputs["consume"]["same"].resolve() is True
                assert result.outputs["consume"]["token"].resolve() == "secret"
                producers = {
                    producer
                    for _rid, _owner, producer, present in value_resource_provenance_refs(
                        result.outputs["produce"]["heavy"]
                    )
                    if present
                }
                assert producers == {"box1"}
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def _dev_pack_variant(tmp_path: Path, name: str, old: str, new: str) -> Path:
    """Copy the dev pack and rewrite one schema fact in the copy; a service
    launched from it (with the copy's src on PYTHONPATH) announces the
    mutated schema while the local composition serves the original."""
    root = tmp_path / name
    shutil.copytree(DEV_MANIFEST.parent, root)
    gallery = root / "src" / "dinkster_nodes_dev" / "gallery.py"
    text = gallery.read_text(encoding="utf-8")
    assert text.count(old) == 1
    gallery.write_text(text.replace(old, new), encoding="utf-8")
    return root


def _dev_pack_with_comfy_registries(
    tmp_path: Path,
) -> tuple[Path, ComfyAliasRegistry, ComfyGroupRegistry]:
    root = tmp_path / "alias-pack"
    shutil.copytree(DEV_MANIFEST.parent, root)
    image = TypeExpr.concrete("dev.image")
    source = ComfyAliasSource("comfy-core", "InvertImage", "comfy.InvertImage", "b78cec87")
    registry = ComfyAliasRegistry(
        source_schemas=(
            ComfyAliasSourceSchema(
                NodeSchema(
                    source.node_type,
                    inputs=(InputSpec("image", image),),
                    outputs=(OutputSpec("image", image),),
                ),
                SCHEMA_WIRE_VERSION,
            ),
        ),
        records=(
            ComfyAliasRecord(
                id="comfy_alias:comfy-core/InvertImage",
                mapping_kind="op",
                carrier="dev.image.invert",
                source=source,
                replacement=ReplacementRule(
                    from_type=source.node_type,
                    cases=(
                        ReplacementCase.build(
                            "dev.image.invert",
                            inputs={"image": MappingSource.copy("image")},
                            outputs={"image": "image"},
                        ),
                    ),
                ),
                confidence=ComfyAliasConfidence("exact", ("tests/test_serve_remote.py",)),
            ),
        ),
    )
    (root / "comfy-aliases.json").write_text(
        json.dumps(comfy_alias_registry_to_wire(registry)),
        encoding="utf-8",
    )
    scale_source = ComfyAliasSource("comfy-core", "ImageScale", "comfy.ImageScale", "b78cec87")
    group_type = "comfy-group.comfy-core.invert-scale"
    group_schema = NodeSchema(
        group_type,
        inputs=(InputSpec("image", image),),
        outputs=(OutputSpec("image", image),),
    )
    group_registry = ComfyGroupRegistry(
        source_schemas=(
            ComfyGroupSourceSchema(
                registry.source_schemas[0].schema,
                registry.source_schemas[0].wire_version,
            ),
            ComfyGroupSourceSchema(
                NodeSchema(
                    scale_source.node_type,
                    inputs=(InputSpec("image", image),),
                    outputs=(OutputSpec("image", image),),
                ),
                SCHEMA_WIRE_VERSION,
            ),
        ),
        group_schemas=(ComfyGroupSourceSchema(group_schema, SCHEMA_WIRE_VERSION),),
        records=(
            ComfyGroupRecord(
                id="comfy_group:comfy-core/invert-scale",
                mapping_kind="op",
                carrier="dev.image.invert",
                source=ComfyGroupSource("comfy-core", "invert-scale", "b78cec87"),
                pattern=ComfyGroupPattern(
                    group_type=group_type,
                    anchor="invert",
                    nodes=(
                        ("invert", ComfyGroupNode(source, "active")),
                        ("scale", ComfyGroupNode(scale_source, "active")),
                    ),
                    edges=(ComfyGroupEdge("invert:image", "scale:image"),),
                    inputs=(("image", "invert:image"),),
                    parameters=(),
                    constants=(),
                    outputs=(("image", "scale:image"),),
                ),
                replacement=ReplacementRule(
                    from_type=group_type,
                    cases=(
                        ReplacementCase.build(
                            "dev.image.invert",
                            inputs={"image": MappingSource.copy("image")},
                            outputs={"image": "image"},
                        ),
                    ),
                ),
                confidence=ComfyAliasConfidence("grouped", ("tests/test_serve_remote.py",)),
            ),
        ),
    )
    (root / "comfy-groups.json").write_text(
        json.dumps(comfy_group_registry_to_wire(group_registry)),
        encoding="utf-8",
    )
    return root, registry, group_registry


def test_remote_composes_comfy_registries_for_selected_owned_nodes(tmp_path: Path) -> None:
    async def scenario() -> None:
        root, aliases, groups = _dev_pack_with_comfy_registries(tmp_path)
        proc, host, port = await start_service(
            root / "dinkster-pack.toml", tmp_path, pythonpath=(root / "src",)
        )
        try:
            composer = ServingComposer()
            try:
                delta = await composer.add_remote(
                    remote_spec(host, port, tmp_path, nodes=("dev.image.invert",))
                )
                assert delta.packs["box1"].comfy_aliases == aliases
                assert delta.packs["box1"].comfy_groups == groups
                assert composer.composition.packs["box1"].comfy_aliases == aliases
                assert composer.composition.packs["box1"].comfy_groups == groups
            finally:
                await composer.close()

            composer = ServingComposer()
            try:
                delta = await composer.add_remote(
                    remote_spec(host, port, tmp_path, nodes=("dev.image.gradient",))
                )
                assert delta.packs["box1"].comfy_aliases is None
                assert delta.packs["box1"].comfy_groups is None
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_presentation_divergent_matching_type_composes_by_signature(tmp_path: Path) -> None:
    """A real daemon announces reduced schemas for executes-claimed types
    (presentation prose stripped), so matching is by schema signature: the
    computational interface, never descriptions or search terms."""
    root = _dev_pack_variant(
        tmp_path,
        "presentation-variant",
        "Rendering surface for every native widget descriptor; ",
        "Reworded prose changing no computational fact whatsoever; ",
    )

    async def scenario() -> None:
        proc, host, port = await start_service(
            root / "dinkster-pack.toml", tmp_path, pythonpath=(root / "src",)
        )
        try:
            composer = ServingComposer()
            try:
                await composer.add_pack(DEV_MANIFEST)
                before_schemas = dict(composer.composition.schemas)
                delta = await composer.add_remote(remote_spec(host, port, tmp_path))
                assert delta.schemas == {}
                assert composer.composition.schemas == before_schemas
                assert [arm.name for arm in composer._topology["dev.gallery.widgets"]] == [
                    "dinkster-nodes-dev",
                    "box1",
                ]
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_computationally_divergent_matching_type_is_refused(tmp_path: Path) -> None:
    root = _dev_pack_variant(
        tmp_path,
        "interface-variant",
        '                    "count",',
        '                    "count_total",',
    )

    async def scenario() -> None:
        proc, host, port = await start_service(
            root / "dinkster-pack.toml", tmp_path, pythonpath=(root / "src",)
        )
        try:
            composer = ServingComposer()
            try:
                await composer.add_pack(DEV_MANIFEST)
                before = composition_snapshot(composer)
                with pytest.raises(CompositionError, match="does not match the schema signature"):
                    await composer.add_remote(remote_spec(host, port, tmp_path))
                assert composition_snapshot(composer) == before
                await daemon_accepts_a_client(host, port)
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_over_local_pack_updates_arms(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            composer = ServingComposer(dev=True)
            try:
                await composer.add_pack(DEV_MANIFEST)
                await composer.add_remote(
                    remote_spec(host, port, tmp_path, nodes=("dev.gallery.widgets",))
                )
                assert [arm.name for arm in composer._topology["dev.gallery.widgets"]] == [
                    "dinkster-nodes-dev",
                    "box1",
                ]
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_provider_routes_an_unrouted_schema_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "schema-owner"
        shutil.copytree(DEV_MANIFEST.parent, owner_root)
        owner_manifest = owner_root / DEV_MANIFEST.name
        owner_manifest.write_text(
            owner_manifest.read_text(encoding="utf-8").replace(
                'namespaces = ["dev"]\n',
                'namespaces = ["dev"]\nschema-only = ["dev.gallery.widgets"]\n',
            ),
            encoding="utf-8",
        )
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_pack(owner_manifest)
                assert not composer._routing.has_route("dev.gallery.widgets")

                delta = await composer.add_remote(
                    remote_spec(host, port, tmp_path, nodes=("dev.gallery.widgets",))
                )
                assert delta.schemas == {
                    "dev.gallery.widgets": composer.composition.schemas["dev.gallery.widgets"]
                }
                assert delta.node_packs == {"dev.gallery.widgets": "dinkster-nodes-dev"}
                assert composer._routing.has_route("dev.gallery.widgets")
                assert [arm.name for arm in composer._topology["dev.gallery.widgets"]] == ["box1"]
                with pytest.raises(CompositionError, match="box1.*dev.gallery.widgets"):
                    await composer.remove_pack("dinkster-nodes-dev")
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_nodes_allowlist_filters_announced_types(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_remote(
                    remote_spec(host, port, tmp_path, nodes=("dev.image.invert",))
                )
                composition = composer.composition
                assert "dev.image.invert" in composition.schemas
                assert "dev.image.gradient" not in composition.schemas
                assert composition.node_packs["dev.image.invert"] == "box1"
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_nodes_allowlist_naming_unannounced_type_refused(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            composer = ServingComposer()
            try:
                before = composition_snapshot(composer)
                with pytest.raises(CompositionError, match="not served by"):
                    await composer.add_remote(
                        remote_spec(host, port, tmp_path, nodes=("dev.image.absent",))
                    )
                assert composition_snapshot(composer) == before
                await daemon_accepts_a_client(host, port)
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


RESERVED_NODES_MODULE = '''\
"""Scaffolding pack announcing a node type under the reserved "dinkster" root."""

from collections.abc import Mapping

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import CORE_STRING

STRING = TypeExpr.concrete(CORE_STRING)


class ReservedEcho(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.remote_test.echo",
            display_name="Reserved Echo",
            category="test",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


NODES = [ReservedEcho]
'''


async def start_reserved_service(
    tmp_path: Path,
) -> tuple[asyncio.subprocess.Process, str, int]:
    """Launch a daemon whose pack announces a reserved-root node type; the
    pack module lives in tmp_path, so it rides PYTHONPATH beside the shared
    test scaffolding."""
    (tmp_path / "reservedpack_nodes.py").write_text(RESERVED_NODES_MODULE, encoding="utf-8")
    manifest = tmp_path / "reserved-pack.toml"
    manifest.write_text(
        '[pack]\nname = "reservedpack"\n\n[pack.entry]\nnodes = "reservedpack_nodes:NODES"\n',
        encoding="utf-8",
    )
    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dinkster_workers.service",
        "--listen",
        "127.0.0.1:0",
        "--manifest",
        str(manifest),
        "--token-file",
        str(token_file),
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(TESTS_DIR)))},
    )
    assert proc.stdout is not None
    try:
        line = (await asyncio.wait_for(proc.stdout.readline(), 60.0)).decode()
    except BaseException:
        proc.kill()
        await proc.wait()
        raise
    assert line.startswith(READY_LINE_PREFIX), f"unexpected ready line: {line!r}"
    endpoint = line[len(READY_LINE_PREFIX) :].strip()
    kind, host, port_text = endpoint.split(":")
    assert kind == "tcp"
    return proc, host, int(port_text)


def test_reserved_root_requires_trust(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_reserved_service(tmp_path)
        try:
            composer = ServingComposer()
            try:
                before = composition_snapshot(composer)
                with pytest.raises(CompositionError, match="reserved root"):
                    await composer.add_remote(remote_spec(host, port, tmp_path))
                assert composition_snapshot(composer) == before
                await daemon_accepts_a_client(host, port)
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_reserved_root_composes_with_trust(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_reserved_service(tmp_path)
        try:
            composer = ServingComposer()
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path, trust_reserved=True))
                assert "dinkster.remote_test.echo" in composer.composition.schemas
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_dead_daemon_fails_atomically(tmp_path: Path) -> None:
    async def scenario() -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        token_file = tmp_path / "remote-token.txt"
        token_file.write_text(TOKEN, encoding="utf-8")
        composer = ServingComposer()
        try:
            before = composition_snapshot(composer)
            with pytest.raises(TransportError, match="could not connect"):
                await composer.add_remote(remote_spec("127.0.0.1", dead_port, tmp_path))
            assert composition_snapshot(composer) == before
            assert composer.composition._isolated == []
        finally:
            await composer.close()

    asyncio.run(scenario())


ARMED_NODES_MODULE = '''\
"""Pack announcing a native body arm for its echo node."""

from collections.abc import Mapping

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import CORE_STRING, TypeRegistry

STRING = TypeExpr.concrete(CORE_STRING)


def register_types(registry: TypeRegistry) -> None:
    pass


class ArmedEcho(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="armed.echo",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str, **kwargs: object) -> Mapping[str, object]:
        return cls.outputs(value="pack:" + value)


class NativeEcho(ArmedEcho):
    @classmethod
    async def execute(cls, *, value: str, **kwargs: object) -> Mapping[str, object]:
        return cls.outputs(value="native:" + value)


NODES = [ArmedEcho]
ARM_NODES = {"native": [NativeEcho]}
'''

ATTENTION_EVIDENCE_MODULE = '''\
"""Stands in for the inference runtime the daemon probes for arm evidence."""

from dinkster_protocol import (
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    derive_attention_route_token,
)


def discover_attention_capabilities():
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", "2.13.0"),),
    )


def discover_attention_route_token(policy="auto"):
    return derive_attention_route_token(
        discover_attention_capabilities(),
        AttentionPolicyConfig(requested_policy=policy),
    )
'''


def write_armed_manifest(root: Path) -> Path:
    """A pack whose manifest declares a native body arm, plus the fake
    inference runtime the daemon needs to mint attention route evidence."""
    root.mkdir()
    (root / "armedpack_nodes.py").write_text(ARMED_NODES_MODULE, encoding="utf-8")
    (root / "dinkster_inference_torch.py").write_text(ATTENTION_EVIDENCE_MODULE, encoding="utf-8")
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "armedpack"\nnamespaces = ["armed"]\n\n'
        '[pack.arms]\nnative = ["armed.echo"]\n\n'
        '[pack.entry]\nnodes = "armedpack_nodes:NODES"\n'
        'types = "armedpack_nodes:register_types"\n'
        'arm_nodes = "armedpack_nodes:ARM_NODES"\n'
    )
    return manifest


def test_remote_body_arm_is_policy_selectable_only_under_placement(tmp_path: Path) -> None:
    """A remote's announced body arms compose as placement-scoped dispatch
    arms: the native policy can pick one when the node is placed on that
    remote, while unhinted planning never sees them."""

    class PrefersNativeArm:
        def __init__(self) -> None:
            self.candidates: list[tuple[str, ...]] = []

        async def select(
            self,
            node_type: str,
            inputs: object,
            arms: tuple[str, object],
            *,
            run_id: str | None = None,
            extension_behavior_hash: str | None = None,
            attention_routes: Mapping[str, AttentionRouteToken | None],
        ) -> ExecutionSelection:
            del node_type, inputs, run_id, extension_behavior_hash
            names = tuple(arms[1])  # type: ignore[call-overload]
            self.candidates.append(names)
            native = next((name for name in names if name.endswith("@native")), None)
            if native is not None:
                return ExecutionSelection(
                    target=native,
                    cache_tag="policy-native",
                    attention_route_token=attention_routes[native],
                )
            return ExecutionSelection(
                target=arms[0],
                cache_tag="policy-default",
                attention_route_token=attention_routes[arms[0]],
            )

    async def scenario() -> None:
        root = tmp_path / "armed"
        manifest = write_armed_manifest(root)
        proc, host, port = await start_service(manifest, tmp_path, pythonpath=(root,))
        try:
            policy = PrefersNativeArm()
            composer = ServingComposer(native_policy=policy)  # type: ignore[arg-type]
            try:
                await composer.add_remote(remote_spec(host, port, tmp_path))
                events: list[EngineEvent] = []
                engine = composer.composition.make_engine(on_event=events.append)
                graph = Graph(nodes={"c": GraphNode("armed.echo", {"value": "hello"})})

                hinted = await engine.run(
                    graph,
                    ["c"],
                    run_id="hinted",
                    execution=composer.place_execution(engine.pin_execution(), {"c": "box1"}),
                )
                assert hinted.outputs["c"]["value"].resolve() == "native:hello"
                assert policy.candidates[-1] == ("box1", "box1@native")
                finished = [
                    event
                    for event in events
                    if event.kind == "node_finished" and event.node_id == "c"
                ]
                assert finished[-1].detail["worker"] == "box1"
                assert finished[-1].detail["executionArm"] == "native"

                unhinted = await engine.run(graph, ["c"], run_id="unhinted")
                assert unhinted.cached == ()
                assert unhinted.outputs["c"]["value"].resolve() == "pack:hello"
                assert policy.candidates[-1] == ("box1",)
            finally:
                await composer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_native_body_arm_without_attention_evidence_is_refused() -> None:
    worker = SimpleNamespace(body_arms={"native": ("armed.echo",)}, attention_route_token=None)
    with pytest.raises(CompositionError, match="attention route evidence"):
        _validated_remote_body_arms("box1", worker, {})


def test_remote_body_arm_naming_unannounced_type_is_refused() -> None:
    worker = SimpleNamespace(body_arms={"alt": ("armed.missing",)}, attention_route_token=None)
    with pytest.raises(CompositionError, match="not announced"):
        _validated_remote_body_arms("box1", worker, {})


def test_remote_body_arms_are_returned_in_deterministic_order() -> None:
    schema = object()
    announced = {"armed.zeta": schema, "armed.alpha": schema, "armed.mid": schema}
    worker = SimpleNamespace(
        body_arms={
            "beta": ("armed.zeta", "armed.alpha"),
            "alt": ("armed.mid", "armed.alpha"),
        },
        attention_route_token=None,
    )
    assert _validated_remote_body_arms("box1", worker, announced) == (  # type: ignore[arg-type]
        ("alt", ("armed.alpha", "armed.mid")),
        ("beta", ("armed.alpha", "armed.zeta")),
    )


def test_merge_remote_budgets_qualifies_devices() -> None:
    budgets = {"ram": 1}
    spec = RemoteSpec(
        name="box",
        host="h",
        port=1,
        token_file=Path("/t"),
        memory_budgets={"ram": 2, "vram:cuda:0": 3},
    )
    merge_remote_budgets(budgets, [spec])
    assert budgets == {"ram": 1, "ram@box": 2, "vram:cuda:0@box": 3}
