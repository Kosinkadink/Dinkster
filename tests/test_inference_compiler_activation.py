"""Graph compiler activation proof over genuine out-of-tree fixture packs."""

# The test intentionally inspects generation/catalog/session state to prove
# lifecycle and transport cleanup rather than replacing any production seam.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_compat_comfy import ResidentPool
from dinkster_graph import Graph, GraphNode
from dinkster_inference import builtin_sampler_snapshot
from dinkster_memory import FullReleaseResult
from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    canonical_compile_reply_bytes,
    extension_behavior_hash,
)
from dinkster_server import STATE_KEY, create_app
from dinkster_workers.doctor import prepare_catalog

from dinkster.compose import CompositionError, PackSpec, ServingComposer
from tests.test_server import submit_body

TESTS = Path(__file__).parent
ATTENTION_PROVIDER = TESTS / "fixtures/attention_provider"
ALPHA = TESTS / "fixtures/compiler_activation_alpha/dinkster-pack.toml"
BETA = TESTS / "fixtures/compiler_activation_beta/dinkster-pack.toml"
GRAPH = {
    "nodes": {
        "source": {
            "nodeType": "std.math.add_ints",
            "inputs": {"a": 1, "b": 2},
        }
    }
}
TARGETS = ("source",)
EXPECTED_ORDER = (
    "compiler_activation_alpha.first",
    "compiler_activation_beta.same_a",
    "compiler_activation_beta.same_b",
)


def _host_manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "sampling-host"\n'
        'namespaces = ["dinkster", "comfy"]\n\n'
        '[pack.arms]\nnative = ["dinkster.ksampler"]\n\n'
        '[pack.entry]\nnodes = "s1_sampler_host:NODES"\n'
        'arm_nodes = "s1_sampler_host:ARM_NODES"\n'
        'choices = "s1_sampler_host:choices"\n',
        encoding="ascii",
    )
    return manifest


def _worker_env(
    *,
    trace: Path | None = None,
    control: Path | None = None,
    cancel_marker: Path | None = None,
    declaration_mode: str = "ok",
) -> dict[str, str]:
    env = {
        "PYTHONPATH": os.pathsep.join(
            (str(ATTENTION_PROVIDER), str(TESTS), str(ALPHA.parent), str(BETA.parent))
        ),
        "DINKSTER_COMPILER_ACTIVATION_DECLARATION_MODE": declaration_mode,
    }
    if trace is not None:
        env["DINKSTER_COMPILER_ACTIVATION_TRACE"] = str(trace)
    if control is not None:
        env["DINKSTER_COMPILER_ACTIVATION_CALLBACK_CONTROL"] = str(control)
    if cancel_marker is not None:
        env["DINKSTER_COMPILER_ACTIVATION_CANCEL_MARKER"] = str(cancel_marker)
    return env


def _digest(runtime: Any) -> str:
    return "sha256:" + extension_behavior_hash(runtime.extension_snapshot)


def _semantic_reply(reply: Mapping[str, object]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        {key: value for key, value in reply.items() if key not in {"type", "requestId"}},
    )


async def _capture_order(root: Path, manifests: tuple[Path, Path]) -> dict[str, object]:
    trace = root / "trace.txt"
    control = root / "control.txt"
    control.parent.mkdir(parents=True)
    control.write_text("ok", encoding="ascii")
    composer = ServingComposer(worker_env=_worker_env(trace=trace, control=control))
    workers: tuple[Any, ...] = ()
    try:
        await composer.add_pack(PackSpec(_host_manifest(root / "host"), trust_reserved=True))
        for manifest in manifests:
            await composer.add_pack(manifest)
        runtime = composer._runtime_seat.pin()
        worker = composer._sampling_worker(composer._topology)
        assert worker is not None
        workers = tuple(composer.composition._isolated)
        transport = runtime.graph_compile_transport
        assert transport is not None
        digest = _digest(runtime)
        replies = []
        for _ in range(2):
            replies.append(_semantic_reply(await transport(digest, GRAPH, TARGETS)))
        replies_tuple = tuple(replies)
        assert worker._session._graph_compile_pending == {}
        return {
            "snapshot": runtime.extension_snapshot,
            "declarations": runtime.graph_compiler_registry.contributions,
            "behavior": extension_behavior_hash(runtime.extension_snapshot),
            "choices": dict(composer.composition.choices),
            "replies": replies_tuple,
            "canonical": tuple(canonical_compile_reply_bytes(reply) for reply in replies_tuple),
            "generated": tuple(
                sorted(set(reply["graph"]["nodes"]) - {"source"})  # type: ignore[index]
                for reply in replies_tuple
            ),
            "origins": tuple(reply["origins"] for reply in replies_tuple),
            "attempts": tuple(reply["attemptedGeneratedCounts"] for reply in replies_tuple),
            "trace": tuple(trace.read_text(encoding="ascii").splitlines()),
        }
    finally:
        await composer.close()
        assert workers and all(not worker.alive for worker in workers)


def test_genuine_packs_are_activation_order_invariant_and_compile_canonically(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        forward = await _capture_order(tmp_path / "forward", (ALPHA, BETA))
        reverse = await _capture_order(tmp_path / "reverse", (BETA, ALPHA))
        for capture in (forward, reverse):
            assert capture["replies"][0] == capture["replies"][1]  # type: ignore[index]
            assert capture["canonical"][0] == capture["canonical"][1]  # type: ignore[index]
            assert capture["generated"][0] == capture["generated"][1]  # type: ignore[index]
            assert capture["origins"][0] == capture["origins"][1]  # type: ignore[index]
            assert capture["attempts"][0] == capture["attempts"][1]  # type: ignore[index]
        assert forward["snapshot"] == reverse["snapshot"]
        assert forward["declarations"] == reverse["declarations"]
        assert forward["behavior"] == reverse["behavior"]
        assert forward["choices"] == reverse["choices"]
        assert forward["replies"] == reverse["replies"]
        assert forward["canonical"] == reverse["canonical"]
        assert forward["generated"] == reverse["generated"]
        assert forward["origins"] == reverse["origins"]
        assert forward["attempts"] == reverse["attempts"] == ([3], [3])
        expected_trace = tuple(
            f"{pass_index}:{compiler_id}"
            for _request in range(2)
            for pass_index in range(2)
            for compiler_id in EXPECTED_ORDER
        )
        assert forward["trace"] == reverse["trace"] == expected_trace
        declarations = cast("tuple[Any, ...]", forward["declarations"])
        assert tuple(item.id for item in declarations) == EXPECTED_ORDER

    asyncio.run(scenario())


@pytest.mark.parametrize("cold", [False, True])
def test_adopted_graph_compile_waits_for_full_free_before_worker_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cold: bool
) -> None:
    host = _host_manifest(tmp_path / "host")
    trace = tmp_path / "trace"
    env = _worker_env(trace=trace)
    extension = shutil.copytree(ALPHA.parent, tmp_path / "extension") / ALPHA.name
    if cold:
        assert prepare_catalog(host, environment=env).ok
        assert prepare_catalog(extension, environment=env).ok

    async def scenario() -> None:
        entered, finish = asyncio.Event(), asyncio.Event()

        class BlockingPool(ResidentPool):
            async def full_release(self) -> FullReleaseResult:
                entered.set()
                await finish.wait()
                return await super().full_release()

        pool = BlockingPool()
        monkeypatch.setattr("dinkster.compose.default_pool", lambda: pool)
        composer = ServingComposer(worker_env=env)
        staged = composer.spawn_empty()
        try:
            await staged.add_pack(PackSpec(host, trust_reserved=True, require_catalog=cold))
            await staged.add_pack(PackSpec(extension, require_catalog=cold))
            async with composer.generation_transaction():
                old = composer.adopt(staged)
            await old.close()
            worker = composer._sampling_worker(composer._topology)
            assert worker is not None
            assert bool(getattr(worker, "cold", False)) == cold
            token = worker.instance_token
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                full_free=composer.full_free,
            )
            async with TestClient(TestServer(app)) as client:
                await client.post("/api/queue/pause")
                release = asyncio.create_task(
                    client.post("/memory/free", json={"requestId": "compile"})
                )
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    graph = Graph(nodes={"source": GraphNode("dinkster.ksampler", {})})
                    submit = asyncio.create_task(
                        client.post("/api/jobs", json=submit_body(graph, ["source"]))
                    )
                    try:
                        done, _ = await asyncio.wait({submit}, timeout=0.1)
                        assert not done and not trace.exists()
                        assert worker.instance_token == token
                        assert bool(getattr(worker, "cold", False)) == cold
                        assert composer._mutate.locked()
                        assert app[STATE_KEY].queue._maintenance_active
                        assert not app[STATE_KEY].queue._running
                        finish.set()
                        response = await release
                        assert response.status == 200
                        body = await response.json()
                        assert body["completed"] is True
                        assert ("sampling-host" in {w["worker"] for w in body["workers"]}) != cold
                        response = await asyncio.wait_for(submit, 5)
                        if cold:
                            assert response.status == 400, await response.json()
                            assert (await response.json())["error"] == "compile-unknown-generation"
                        else:
                            assert response.status == 202, await response.json()
                            assert trace.exists()
                        assert worker.alive
                        assert not getattr(worker, "cold", False)
                        assert app[STATE_KEY].queue.paused
                    finally:
                        finish.set()
                        await submit
                finally:
                    finish.set()
                    await release
        finally:
            finish.set()
            await composer.close()
            if staged._private_staging:
                await staged.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("grouped", [False, True])
def test_warm_compiler_waits_for_child_pool_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouped: bool
) -> None:
    host = _host_manifest(tmp_path / "host")
    with host.open("a", encoding="ascii") as stream:
        stream.write('consumers = "full_free_compiler:consumers"\n')
    extension = tmp_path / "extension.toml"
    extension.write_text(
        '[pack]\nname = "maintenance_compiler"\n'
        '[pack.entry]\nnodes = "s1_sampler_empty:NODES"\n'
        '[pack.extension]\ninference = "full_free_compiler:register"\n'
        'privileges = ["inference"]\n',
        encoding="ascii",
    )
    environment = {**_worker_env(), "DINKSTER_FULL_FREE_COMPILER_ROOT": str(tmp_path)}

    async def scenario() -> None:
        pool = ResidentPool()
        monkeypatch.setattr("dinkster.compose.default_pool", lambda: pool)
        composer = ServingComposer(worker_env=environment)
        finish = tmp_path / "finish"
        try:
            for path in (host, extension):
                await composer.add_pack(
                    PackSpec(
                        path,
                        trust_reserved=path == host,
                        aimdo="off",
                        worker_group="compiler-group" if grouped else None,
                        group_manifests=(host, extension) if grouped else (),
                    )
                )
            worker = composer._records["sampling-host"].worker
            token = worker.instance_token
            assert worker.alive and token is not None
            app = create_app(
                composer.composition.make_engine,
                composer.composition.schemas,
                full_free=composer.full_free,
            )
            async with TestClient(TestServer(app)) as client:
                await client.post("/api/queue/pause")
                release = asyncio.create_task(
                    client.post("/memory/free", json={"requestId": "warm-compiler"})
                )
                try:
                    async with asyncio.timeout(5):
                        while not (tmp_path / "commit-entered").exists():
                            await asyncio.sleep(0.005)
                    graph = Graph(nodes={"sample": GraphNode("dinkster.ksampler", {})})
                    submit = asyncio.create_task(
                        client.post("/api/jobs", json=submit_body(graph, ["sample"]))
                    )
                    try:
                        done, _ = await asyncio.wait({submit}, timeout=0.1)
                        assert not done and not (tmp_path / "compiled").exists()
                        assert not (tmp_path / "commit-finished").exists()
                        assert composer._mutate.locked()
                        assert app[STATE_KEY].queue._maintenance_active
                        assert not app[STATE_KEY].queue._running
                        assert worker.alive and worker.instance_token == token
                        finish.touch()
                        result = await (await release).json()
                        assert result["completed"] is True
                        sampled = next(
                            item for item in result["workers"] if item["worker"] == "sampling-host"
                        )
                        assert sampled["consumers"] == [{"consumer": "pool", "status": "complete"}]
                        response = await asyncio.wait_for(submit, 5)
                        assert response.status == 202, await response.json()
                        assert (tmp_path / "compiled").exists()
                        assert worker.alive and worker.instance_token == token
                        assert app[STATE_KEY].queue.paused
                    finally:
                        finish.touch()
                        await submit
                finally:
                    finish.touch()
                    await release
        finally:
            finish.touch()
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("declaration_mode", ("duplicate", "collision"))
def test_colliding_compiler_declarations_publish_nothing(
    tmp_path: Path, declaration_mode: str
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(declaration_mode=declaration_mode))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(ALPHA)
            published = composer._runtime_seat.pin()
            choices = dict(composer.composition.choices)
            catalog = composer._sampler_catalog_path.read_bytes()
            with pytest.raises(CompositionError, match="activation failed"):
                await composer.add_pack(BETA)
            assert composer._runtime_seat.pin() is published
            assert dict(composer.composition.choices) == choices
            assert composer._sampler_catalog_path.read_bytes() == catalog
            assert "compiler_activation_beta" not in composer.composition.packs
            worker = composer._sampling_worker(composer._topology)
            assert worker is not None and worker._session._graph_compile_pending == {}
        finally:
            workers = tuple(composer.composition._isolated)
            await composer.close()
            assert workers and all(not worker.alive for worker in workers)

    asyncio.run(scenario())


def test_real_compile_failure_cancellation_rotation_and_unload_are_transactional(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        control = tmp_path / "control.txt"
        cancel_marker = tmp_path / "cancel-entered.txt"
        control.write_text("ok", encoding="ascii")
        composer = ServingComposer(
            worker_env=_worker_env(control=control, cancel_marker=cancel_marker)
        )
        worker = None
        workers: tuple[Any, ...] = ()
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            baseline = composer._runtime_seat.pin()
            baseline_choices = dict(composer.composition.choices)
            baseline_behavior = extension_behavior_hash(baseline.extension_snapshot)
            assert baseline.sampler_registry_snapshot == builtin_sampler_snapshot()
            assert baseline.graph_compile_transport is None

            await composer.add_pack(ALPHA)
            old_runtime = composer._runtime_seat.pin()
            old_digest = _digest(old_runtime)
            old_transport = old_runtime.graph_compile_transport
            assert old_transport is not None
            old_record = json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))[
                "records"
            ][old_digest]

            await composer.add_pack(BETA)
            current = composer._runtime_seat.pin()
            current_digest = _digest(current)
            transport = current.graph_compile_transport
            assert transport is not None and current_digest != old_digest
            worker = composer._sampling_worker(composer._topology)
            assert worker is not None
            workers = tuple(composer.composition._isolated)

            old_reply = _semantic_reply(await old_transport(old_digest, GRAPH, TARGETS))
            assert old_reply["generationKey"] == old_digest
            with pytest.raises(AssertionError, match="owning digest"):
                await old_transport(current_digest, GRAPH, TARGETS)

            before_refusal = composer._sampler_catalog_path.read_bytes()
            with pytest.raises(CompositionError, match="retired.*cannot be.*activated"):
                await composer.remove_pack("compiler_activation_beta")
            assert composer._runtime_seat.pin() is current
            assert composer._sampler_catalog_path.read_bytes() == before_refusal
            assert json.loads(before_refusal)["records"][old_digest] == old_record

            control.write_text("failure", encoding="ascii")
            failed = await transport(current_digest, GRAPH, TARGETS)
            assert failed["errorName"] == GRAPH_COMPILE_ERROR_COMPILER_FAILURE
            error = cast(str, failed["error"])
            assert "compiler_activation_beta.same_a" in error
            assert "compiler-activation-beta-callback-failure" in error
            assert worker._session._graph_compile_pending == {}

            control.write_text("cancel", encoding="ascii")

            async def compile_current() -> Mapping[str, object]:
                return await transport(current_digest, GRAPH, TARGETS)

            cancelled = asyncio.create_task(compile_current())
            for _ in range(1000):
                if (
                    cancel_marker.exists()
                    and cancel_marker.read_text(encoding="ascii") == "entered"
                ):
                    break
                await asyncio.sleep(0.005)
            else:
                pytest.fail("cancel compiler did not reach its cancellation checkpoint")
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert worker._session._graph_compile_pending == {}
            for _ in range(1000):
                if cancel_marker.read_text(encoding="ascii") == "cancelled":
                    break
                await asyncio.sleep(0.005)
            else:
                pytest.fail("cancel compiler did not observe cancellation")

            control.write_text("ok", encoding="ascii")
            retried = _semantic_reply(await transport(current_digest, GRAPH, TARGETS))
            assert retried["attemptedGeneratedCounts"] == [3]
            assert worker._session._graph_compile_pending == {}

            await composer.remove_pack("compiler_activation_alpha")
            await composer.remove_pack("compiler_activation_beta")
            restored = composer._runtime_seat.pin()
            assert restored.extension_snapshot == baseline.extension_snapshot
            assert restored.sampler_registry_snapshot == baseline.sampler_registry_snapshot
            assert restored.graph_compiler_registry == baseline.graph_compiler_registry
            assert restored.graph_compile_transport is baseline.graph_compile_transport is None
            assert dict(composer.composition.choices) == baseline_choices
            assert extension_behavior_hash(restored.extension_snapshot) == baseline_behavior
            retired_catalog = json.loads(
                composer._sampler_catalog_path.read_text(encoding="utf-8")
            )["records"]
            assert retired_catalog[old_digest] == old_record
            assert retired_catalog
            assert _semantic_reply(await old_transport(old_digest, GRAPH, TARGETS)) == old_reply
        finally:
            await composer.close()
            assert workers and all(not item.alive for item in workers)
            if worker is not None:
                assert worker._session._graph_compile_pending == {}

    asyncio.run(scenario())
