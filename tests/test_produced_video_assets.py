"""VIDEO sources outlive the worker and the transient transfer cache."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import shutil
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_api.v1 import ASSET_TYPE, AUDIO_TYPE, VIDEO_TYPE
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetResolver,
    AssetVault,
    digest_bytes,
    register_audio_value_type,
    register_video_value_type,
    resolver_from_env,
)
from dinkster_assets._sqlite import connect_journal
from dinkster_assets.declared import produced_output_scratch, use_produced_output_job
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchState,
    claim_inactive_produced_job,
    claim_inactive_produced_scratch,
    hold_produced_job,
)
from dinkster_caches import BudgetedDiskCAS
from dinkster_engine.control import EngineRunContext, RunControlState
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import Invocation
from dinkster_schema import Node, NodeSchema, RunExecutionContext
from dinkster_values import TypeRegistry, Value, ValueMeta, make_list_value
from dinkster_values.model import PyObjPayload
from dinkster_workers import BoundaryDiagnostic, InProcessWorker, IsolatedWorker, RemoteWorker
from dinkster_workers import session as session_module
from dinkster_workers.boundary import read_frame
from dinkster_workers.produced_assets import (
    ProducedAssetAuthority,
    source_asset_files,
)
from produced_video_nodes import ProduceAudio, ProduceVideo
from test_remote import TOKEN, start_service, stop_service

from tests.async_scenarios import run_scenario


@pytest.fixture
def vaults(tmp_path: Path) -> tuple[AssetVault, AssetVault]:
    return AssetVault(tmp_path / "producer"), AssetVault(tmp_path / "engine")


def _publish(vault: AssetVault, data: bytes) -> AssetRef:
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return AssetRef(digest, "source.mkv", len(data), resolver=vault)


def _open_new_journals(channel: Connection, root: Path, barrier: Barrier) -> None:
    results = []
    try:
        for index in range(32):
            barrier.wait(timeout=60)
            try:
                with closing(connect_journal(root / f"journal-{index}.sqlite")) as db:
                    results.append(db.execute("PRAGMA journal_mode").fetchone())
            except sqlite3.Error as exc:
                results.append(repr(exc))
        channel.send(results)
    finally:
        channel.close()


def _publish_with_database_lock(channel: Connection, root: Path, hold: bool) -> None:
    state = ProducedScratchState(root)
    data = b"lock holder" if hold else b"contending publisher"
    try:
        if hold:
            state.published(digest_bytes(data), len(data))
            with sqlite3.connect(root / ".produced-scratch.sqlite") as db:
                db.execute("BEGIN IMMEDIATE")
                channel.send("locked")
                assert channel.poll(60), "lock holder was not released"
                assert channel.recv() == "release"
        else:
            channel.send("publishing")
            state.published(digest_bytes(data), len(data))
            state.offer((digest_bytes(data),))
        channel.send(None)
    except Exception as exc:
        channel.send(repr(exc))
    finally:
        channel.close()


def _hold_produced_scratch(channel: Connection, env: dict[str, str]) -> None:
    job_lease = hold_produced_job(Path(env["DINKSTER_PACK_SCRATCH"]), "process job")
    with use_produced_output_job("process job", 1):
        vault = produced_output_scratch(env)
        assert vault is not None
        _publish(vault, b"recoverable publication")
        writer = vault.writer(digest_bytes(b"pending bytes"))
        writer.write(b"pending bytes")
        channel.send(str(vault.root))
        command = channel.recv()
        if command == "crash":
            os._exit(0)
        assert command == "close scope"
    channel.send("scope closed")
    assert channel.recv() == "close writer"
    writer.__exit__(None, None, None)
    channel.send("writer closed")
    job_lease.close()
    channel.close()


@pytest.mark.parametrize("crash", [False, True])
def test_produced_scratch_claim_protects_live_process_and_open_writer(
    tmp_path: Path, crash: bool
) -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    env = {"DINKSTER_PACK_SCRATCH": str(tmp_path / "scratch")}
    process = context.Process(target=_hold_produced_scratch, args=(child, env))
    process.start()
    child.close()

    def receive() -> object:
        assert parent.poll(20), "producer did not complete the requested transition"
        return parent.recv()

    try:
        root = Path(cast(str, receive()))
        with claim_inactive_produced_job(Path(env["DINKSTER_PACK_SCRATCH"]), "process job") as idle:
            assert not idle
        with claim_inactive_produced_scratch(root) as inactive:
            assert not inactive
        parent.send("crash" if crash else "close scope")
        if not crash:
            assert receive() == "scope closed"
            with claim_inactive_produced_job(
                Path(env["DINKSTER_PACK_SCRATCH"]), "process job"
            ) as idle:
                assert not idle
            with claim_inactive_produced_scratch(root) as inactive:
                assert not inactive
            parent.send("close writer")
            assert receive() == "writer closed"
        process.join(20)
        assert process.exitcode == 0
        with claim_inactive_produced_job(Path(env["DINKSTER_PACK_SCRATCH"]), "process job") as idle:
            assert idle
        assert ProducedScratchState(root).publications() == (
            (digest_bytes(b"recoverable publication"), len(b"recoverable publication"), False),
        )
        with claim_inactive_produced_scratch(root) as inactive:
            assert inactive
            with use_produced_output_job("process job", 1):
                with pytest.raises(AssetError, match="busy with reconciliation"):
                    produced_output_scratch(env)
        with use_produced_output_job("process job", 1):
            vault = produced_output_scratch(env)
            assert vault is not None
            assert _publish(vault, b"next invocation").read_bytes() == b"next invocation"
        path = AssetVault(root).resolve(digest_bytes(b"recoverable publication"))
        assert path is not None
        assert path.read_bytes() == b"recoverable publication"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(20)
        parent.close()


def test_produced_scratch_isolates_concurrent_jobs_and_attempts(tmp_path: Path) -> None:
    env = {"DINKSTER_PACK_SCRATCH": str(tmp_path / "scratch")}
    resolver = resolver_from_env(env)
    assert resolver is not None
    data = b"identical produced bytes"
    digest = digest_bytes(data)

    async def scenario() -> Sequence[Path]:
        ready = [asyncio.Event() for _ in range(3)]

        async def publish(index: int, job: str, attempt: int) -> Path:
            with use_produced_output_job(job, attempt):
                vault = produced_output_scratch(env)
                assert vault is not None
                assert resolver.resolve(digest) is None
                ref = await asyncio.to_thread(_publish, vault, data)
                ready[index].set()
                await asyncio.gather(*(event.wait() for event in ready))
                assert resolver.resolve(digest) == ref.local_path()
                assert await asyncio.to_thread(ref.read_bytes) == data
                return ref.local_path()

        return await asyncio.wait_for(
            asyncio.gather(
                publish(0, "../job with spaces", 1),
                publish(1, "../job with spaces", 2),
                publish(2, "another job", 1),
            ),
            10,
        )

    paths = run_scenario(scenario())
    assert len(set(paths)) == 3
    assert all(
        path.is_relative_to(tmp_path / "scratch" / "produced-assets" / "jobs") for path in paths
    )
    assert resolver.resolve(digest) is None
    with use_produced_output_job("../job with spaces", 1):
        assert resolver.resolve(digest) == paths[0]
    assert not (tmp_path / "job with spaces").exists()


@pytest.mark.parametrize("outcome", ["complete", "failure", "cancel"])
@pytest.mark.parametrize("store_kind", ["scratch", "vault"])
def test_completed_produced_scope_refuses_detached_tasks_and_held_writers(
    tmp_path: Path, outcome: str, store_kind: str
) -> None:
    env = {
        "DINKSTER_PACK_SCRATCH" if store_kind == "scratch" else "DINKSTER_ASSET_VAULT": str(
            tmp_path / store_kind
        )
    }
    data = b"unpublished source"
    digest = digest_bytes(data)

    async def scenario() -> None:
        resume = asyncio.Event()
        detached: list[asyncio.Task[None]] = []

        async def owner() -> None:
            with use_produced_output_job("ending job", 1) as owns:
                assert owns
                vault = produced_output_scratch(env)
                assert vault is not None
                writer = vault.writer(digest)
                writer.write(data)

                async def late_publication() -> None:
                    await resume.wait()
                    try:
                        with pytest.raises(AssetError, match="no longer active"):
                            produced_output_scratch(env)
                        with pytest.raises(AssetError, match="no longer active"):
                            vault.writer(digest)
                        with pytest.raises(AssetError, match="no longer active"):
                            writer.write(b"late bytes")
                        with pytest.raises(AssetError, match="no longer active"):
                            writer.commit()
                        with pytest.raises(AssetError, match="no longer active"):
                            writer.commit_with_result()
                    finally:
                        writer.__exit__(None, None, None)

                detached.append(asyncio.create_task(late_publication()))
                if outcome == "failure":
                    raise ValueError("job failed")
                if outcome == "cancel":
                    raise asyncio.CancelledError

        try:
            await owner()
        except (ValueError, asyncio.CancelledError):
            assert outcome != "complete"
        resume.set()
        await asyncio.wait_for(asyncio.gather(*detached), 5)
        with use_produced_output_job("ending job", 1):
            vault = produced_output_scratch(env)
            assert vault is not None
            assert vault.resolve(digest) is None
            assert not list(vault.root.glob(".ingest-*"))
            with use_produced_output_job("ending job", 1) as owns:
                assert not owns
            assert _publish(vault, data).read_bytes() == data

    run_scenario(scenario())


@pytest.mark.parametrize("configured_vault", [False, True])
@pytest.mark.parametrize("media", ["audio", "video"])
def test_in_process_produced_sources_require_durable_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_vault: bool, media: str
) -> None:
    engine = AssetVault(tmp_path / "engine")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(scratch))
    if configured_vault:
        monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(engine.root))
    else:
        monkeypatch.delenv("DINKSTER_ASSET_VAULT", raising=False)
    registry = TypeRegistry()
    register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    node = ProduceVideo if media == "video" else ProduceAudio
    node_type = node.define_schema().node_type
    worker = InProcessWorker({node_type: node}, registry)

    async def scenario() -> None:
        result = await worker.invoke(
            Invocation(
                invocation_id="produce",
                job_ref="local job",
                attempt_id=3,
                node_id="produce",
                node_type=node_type,
                inputs={},
                effective_schema=worker.schemas[node_type],
            )
        )
        with use_produced_output_job("local job", 3):
            store = produced_output_scratch()
            assert store is not None
            digests = store.digests()
            assert len(digests) == 1
            produced = store.resolve(digests[0])
            assert produced is not None
            original = produced.read_bytes()
            assert digest_bytes(original) == digests[0]
            assert ProducedScratchState(store.root).publications() == (
                (digests[0], len(original), True),
            )
        if not configured_vault:
            assert result.error is not None
            assert "require an engine DINKSTER_ASSET_VAULT" in result.error.message
            assert produced.read_bytes() == original
            assert not engine.digests()
            return
        assert result.error is None, result.error
        assert result.outputs is not None
        shutil.rmtree(scratch)
        value = cast(Mapping[str, object], result.outputs[media].resolve())
        source = cast(AssetRef, value["source"])
        assert source.digest == digests[0]
        assert source.local_path() == engine.resolve(digests[0])
        assert source.read_bytes() == original

    run_scenario(scenario())


@pytest.mark.parametrize("configured_vault", [False, True])
@pytest.mark.parametrize("media", ["audio", "video"])
def test_in_process_control_sources_are_adopted_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_vault: bool, media: str
) -> None:
    engine = AssetVault(tmp_path / "engine")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(scratch))
    if configured_vault:
        monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(engine.root))
    else:
        monkeypatch.delenv("DINKSTER_ASSET_VAULT", raising=False)
    registry = TypeRegistry()
    register_video_value_type(registry, VIDEO_TYPE, resolver_from_env())
    register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())
    producer = ProduceVideo if media == "video" else ProduceAudio
    original: dict[str, bytes] = {}
    saved_contexts: list[RunExecutionContext] = []

    class PublishMedia(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema("control.media", outputs=(), execution_context=True)

        @classmethod
        async def execute(cls, execution_context: RunExecutionContext) -> Mapping[str, object]:
            saved_contexts.append(execution_context)
            value = registry.wrap(
                VIDEO_TYPE if media == "video" else AUDIO_TYPE, producer.execute()[media]
            )
            store = produced_output_scratch()
            assert store is not None
            for digest in store.digests():
                path = store.resolve(digest)
                assert path is not None
                original[digest] = path.read_bytes()
            assert len(original) == 1
            await execution_context.publish_run_value("target", {media: value})
            return {}

    worker = InProcessWorker({"control.media": PublishMedia}, registry)

    async def scenario() -> None:
        state = RunControlState()

        def published(_target: str, values: Mapping[str, Value]) -> None:
            digest, data = next(iter(original.items()))
            assert digest_bytes(data) == digest
            path = engine.resolve(digest)
            assert path is not None, "publication reached the engine before durable adoption"
            assert path.read_bytes() == data
            with use_produced_output_job("control job", 2):
                store = produced_output_scratch()
                assert store is not None
                assert ProducedScratchState(store.root).publications() == (
                    (digest, len(data), True),
                )
            # The invocation still owns lease files outside its producer data.
            shutil.rmtree(store.root)
            assert not store.root.exists()
            payload = cast(Mapping[str, object], values[media].resolve())
            assert cast(AssetRef, payload["source"]).read_bytes() == data

        context = EngineRunContext(
            "publish",
            {},
            state,
            lambda: Graph({"target": GraphNode("control.media")}),
            published,
            lambda _nodes: None,
            lambda _node, _inputs: None,
            {},
        )
        result = await worker.invoke(
            Invocation(
                invocation_id="control-source",
                job_ref="control job",
                attempt_id=2,
                node_id="publish",
                node_type="control.media",
                inputs={},
                effective_schema=worker.schemas["control.media"],
                run_control=context,
            )
        )
        if configured_vault:
            assert result.error is None, result.error
            assert state.future("target").done()
            values = await context.await_run_value("target")
            payload = cast(Mapping[str, object], values[media].resolve())
            assert cast(AssetRef, payload["source"]).read_bytes() == next(iter(original.values()))
        else:
            assert result.error is not None
            assert "require an engine DINKSTER_ASSET_VAULT" in result.error.message
            assert not state.future("target").done()
            with use_produced_output_job("control job", 2):
                store = produced_output_scratch()
                assert store is not None
                digest, data = next(iter(original.items()))
                path = store.resolve(digest)
                assert path is not None
                assert path.read_bytes() == data
        with pytest.raises(RuntimeError, match="no longer active"):
            await saved_contexts[0].publish_run_value("target", {})
        state.close()

    run_scenario(scenario())


def _value(ref: AssetRef, type_id: str = VIDEO_TYPE) -> Value:
    return Value(
        type_id=type_id,
        fingerprint="source",
        meta=ValueMeta({"asset_refs": [ref.to_wire()]}),
        payload=PyObjPayload({"source": ref, "edits": []}),
    )


@pytest.mark.parametrize("type_id", [VIDEO_TYPE, "dinkster.layers", AUDIO_TYPE, "custom.asset"])
def test_adoption_survives_producer_removal_and_handles_list_values(
    vaults: tuple[AssetVault, AssetVault],
    type_id: str,
) -> None:
    producer, engine = vaults
    data = b"encoded-source" * 100_000
    ref = _publish(producer, data)
    digest = ref.digest
    value = _value(ref, type_id)
    outputs = {"clips": make_list_value(type_id, (value, value))}
    assert source_asset_files(outputs, producer) == [(digest, ref.local_path())]
    authority = ProducedAssetAuthority(engine)
    authority.capture(outputs, producer)
    ref.local_path().unlink()
    authority.capture(outputs)
    landed = engine.resolve(digest)
    assert landed is not None
    assert landed.read_bytes() == data


@pytest.mark.parametrize("declared_size", [2, 4])
def test_adoption_rejects_size_mismatch_without_publishing(
    vaults: tuple[AssetVault, AssetVault], declared_size: int
) -> None:
    producer, engine = vaults
    digest = _publish(producer, b"123").digest
    authority = ProducedAssetAuthority(engine, producer)
    with pytest.raises(AssetError, match="byte size|truncated"):
        authority.capture({"video": _value(AssetRef(digest, "source", declared_size))})
    assert engine.resolve(digest) is None
    authority.capture({"video": _value(AssetRef(digest, "source", 3))})
    adopted = ProducedAssetInventory(engine.root).entries()
    assert len(adopted) == 1 and adopted[0].size == 3
    assert adopted[0].adopted_at_ns is not None
    assert AssetRef(digest, "source", 3, resolver=engine).read_bytes() == b"123"


def test_adoption_rejects_corrupt_content_and_missing_sources(
    vaults: tuple[AssetVault, AssetVault],
) -> None:
    producer, engine = vaults
    digest = _publish(producer, b"good").digest
    path = producer.resolve(digest)
    assert path is not None
    path.write_bytes(b"evil")
    authority = ProducedAssetAuthority(engine, producer)
    outputs = {"video": _value(AssetRef(digest, "source", 4))}
    with pytest.raises(AssetError, match="ingest did not verify"):
        authority.capture(outputs)
    assert engine.resolve(digest) is None
    path.unlink()
    with pytest.raises(AssetError, match="did not reach the engine"):
        authority.capture(outputs)


def test_produced_adoption_inventory_survives_restart_and_failed_copy(
    vaults: tuple[AssetVault, AssetVault],
) -> None:
    producer, engine = vaults
    ref = _publish(producer, b"recoverable original")
    source = ref.local_path()
    source.write_bytes(b"incorrect source----")
    outputs = {"video": _value(ref)}
    with pytest.raises(AssetError, match="ingest did not verify"):
        ProducedAssetAuthority(engine, producer).capture(outputs)
    pending = ProducedAssetInventory(engine.root).entries()
    assert [(row.digest, row.size, row.adopted_at_ns) for row in pending] == [
        (ref.digest, ref.size, None)
    ]
    assert engine.resolve(ref.digest) is None
    assert source.read_bytes() == b"incorrect source----"
    source.write_bytes(b"recoverable original")
    ProducedAssetAuthority(engine, producer).capture(outputs)
    adopted = ProducedAssetInventory(engine.root).entries()
    assert len(adopted) == 1
    assert adopted[0].digest == ref.digest
    assert adopted[0].size == ref.size
    assert adopted[0].adopted_at_ns is not None
    source.unlink()
    ProducedAssetAuthority(AssetVault(engine.root)).capture(outputs)
    assert ProducedAssetInventory(engine.root).entries() == adopted
    assert replace(ref, resolver=engine).read_bytes() == b"recoverable original"


@pytest.mark.parametrize("type_id", [VIDEO_TYPE, "dinkster.layers", AUDIO_TYPE, "custom.asset"])
@pytest.mark.parametrize("asset_input", [None, ASSET_TYPE, f"asset<{VIDEO_TYPE}>"])
def test_returned_input_sources_need_no_produced_output_scratch(
    type_id: str, asset_input: str | None
) -> None:
    class Unreadable:
        transport = "pyobj"

        def load(self) -> object:
            raise AssertionError("source ownership must not decode payloads")

    ref = AssetRef(digest_bytes(b"caller-owned"), "source", 12)
    value = Value(type_id, "source", ValueMeta({"asset_refs": [ref.to_wire()]}), Unreadable())
    source = (
        Value(asset_input, ref.digest, ValueMeta(ref.to_wire()), Unreadable())
        if asset_input
        else value
    )
    inputs = {"sources": make_list_value(source.type_id, (source,))}
    outputs = {"edited": value}
    authority = ProducedAssetAuthority(None)
    authority.capture(outputs, inputs=inputs)
    with pytest.raises(AssetError, match="require an engine DINKSTER_ASSET_VAULT"):
        authority.capture(outputs)
    wrong_size = _value(AssetRef(ref.digest, ref.name, ref.size + 1), type_id)
    with pytest.raises(AssetError, match="disagree about byte size"):
        authority.capture({"edited": wrong_size}, inputs=inputs)
    produced = _value(AssetRef(digest_bytes(b"new"), "new", 3), type_id)
    with pytest.raises(AssetError, match="require an engine DINKSTER_ASSET_VAULT"):
        authority.capture({"existing": value, "produced": produced}, inputs=inputs)


@pytest.fixture
def producer_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "vproducer"\npublisher = "dinkster"\nnamespaces = ["vproducer"]\n'
        '[pack.entry]\nnodes = "produced_video_nodes:NODES"\n'
        'types = "produced_video_nodes:register_types"\n'
    )
    return manifest


@pytest.mark.parametrize("remote", [False, True], ids=["shared-memory", "remote"])
@pytest.mark.parametrize("adoption_failure", [False, True], ids=["durable", "failed-adoption"])
def test_produced_video_survives_shutdown_and_transfer_eviction(
    producer_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote: bool,
    adoption_failure: bool,
) -> None:
    engine = AssetVault(tmp_path / "engine")
    producer = tmp_path / "producer"
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(engine.root))
    registry = TypeRegistry()
    register_video_value_type(registry, VIDEO_TYPE, engine)
    store = BudgetedDiskCAS(tmp_path / "transfer")
    diagnostics: list[BoundaryDiagnostic] = []

    async def scenario() -> None:
        proc = None
        worker: IsolatedWorker | RemoteWorker
        if remote:
            proc, host, port = await start_service(
                producer_manifest,
                tmp_path,
                "--asset-vault",
                str(producer),
                "--value-store",
                str(tmp_path / "daemon-store"),
            )
            worker = RemoteWorker(
                host,
                port,
                TOKEN,
                registry,
                name="producer",
                value_store=store,
                on_diagnostic=diagnostics.append,
            )
        else:
            worker = IsolatedWorker(
                producer_manifest,
                registry,
                shm_threshold=64,
                extra_env={
                    "PYTHONPATH": str(Path(__file__).parent),
                    "DINKSTER_PACK_SCRATCH": str(producer),
                },
                on_diagnostic=diagnostics.append,
            )
        try:
            await worker.start()
            authority = worker._session._produced_assets
            capture = authority.capture
            if adoption_failure:

                def fail(*args: object) -> None:
                    raise AssetError("injected adoption failure")

                monkeypatch.setattr(authority, "capture", fail)
            result = await worker.invoke(
                Invocation(
                    invocation_id="produce",
                    node_id="produce",
                    node_type="vproducer.make",
                    inputs={},
                    effective_schema=worker.schemas["vproducer.make"],
                )
            )
            if adoption_failure:
                assert result.error is not None
                assert "injected adoption failure" in result.error.message
                monkeypatch.setattr(authority, "capture", capture)
                result = await worker.invoke(
                    Invocation(
                        invocation_id="retry",
                        node_id="produce",
                        node_type="vproducer.make",
                        inputs={},
                        effective_schema=worker.schemas["vproducer.make"],
                    )
                )
            assert result.error is None, result.error
            assert result.outputs is not None
            assert not worker._session._received_asset_pins
            value = result.outputs["video"]
            refs = cast(list[dict[str, object]], value.meta.get("asset_refs"))
            assert len(refs) == 1
            ref = AssetRef.from_wire(refs[0], engine)
            assert ref.size > 256 * 1024
            assert engine.has(ref.digest)
            if remote and not adoption_failure:
                edge = diagnostics[0].outputs[0]
                assert edge.network_bytes >= ref.size
            if not remote:
                assert diagnostics[-1].outputs[0].transport == "shm"
            await worker.close()
            if proc is not None:
                await stop_service(proc)
            shutil.rmtree(producer)
            shutil.rmtree(store.root)
            # Decode only after the producing process and both transient stores are gone.
            video = cast(Mapping[str, object], value.resolve())
            source = cast(AssetRef, video["source"])
            assert source.local_path() == ref.local_path()
            assert digest_bytes(source.read_bytes()) == ref.digest
        finally:
            await worker.close()
            if proc is not None:
                await stop_service(proc)

    run_scenario(scenario())


@pytest.mark.parametrize("cancel", [False, True], ids=["resume", "repeated-cancel"])
def test_produced_video_adoption_owns_source_until_finished(
    producer_manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    engine = AssetVault(tmp_path / "engine")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(engine.root))
    registry = TypeRegistry()
    register_video_value_type(registry, VIDEO_TYPE, engine)
    store = BudgetedDiskCAS(tmp_path / "transfer", max_bytes=1)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    captured: dict[str, Value] = {}

    async def scenario() -> None:
        proc, host, port = await start_service(
            producer_manifest,
            tmp_path,
            "--asset-vault",
            str(tmp_path / "producer"),
            "--value-store",
            str(tmp_path / "daemon-store"),
        )
        worker = RemoteWorker(
            host,
            port,
            TOKEN,
            registry,
            name="producer",
            value_store=store,
            engine_instance_id="video-lifetime",
        )
        session = worker._session
        capture = session._produced_assets.capture
        send = session.send
        acks: list[str] = []

        def blocked_capture(
            outputs: Mapping[str, Value],
            transferred: AssetResolver | None = None,
            inputs: Mapping[str, Value] | None = None,
        ) -> None:
            captured.update(outputs)
            entered.set()
            assert release.wait(15)
            capture(outputs, transferred, inputs)
            finished.set()

        async def record_send(
            header: Mapping[str, object],
            blobs: Sequence[bytes],
            segments: Sequence[SharedMemory] = (),
        ) -> None:
            if header.get("type") in ("resultAck", "cancel"):
                acks.append(str(header["type"]))
            await send(header, blobs, segments)

        monkeypatch.setattr(session._produced_assets, "capture", blocked_capture)
        monkeypatch.setattr(session, "send", record_send)
        run = None
        closing = None
        try:
            await worker.start()
            run = asyncio.create_task(
                worker.invoke(
                    Invocation(
                        invocation_id="produce",
                        node_id="produce",
                        node_type="vproducer.make",
                        inputs={},
                        effective_schema=worker.schemas["vproducer.make"],
                    )
                )
            )
            assert await asyncio.to_thread(entered.wait, 10)
            refs = cast(list[dict[str, object]], captured["video"].meta.get("asset_refs"))
            ref = AssetRef.from_wire(refs[0], engine)
            assert not engine.has(ref.digest)
            if cancel:
                for _ in range(2):
                    run.cancel()
                    await asyncio.sleep(0)
                closing = asyncio.create_task(worker.close())
                done, _ = await asyncio.wait({closing}, timeout=0.05)
                assert not done, "close must drain the still-running capture thread"
                assert session._asset_adoptions
                assert all(not task.done() for task in session._asset_adoptions)
            else:
                await session.detach_transport()
                replay_processed = asyncio.Event()
                replay_seen = False

                async def observe_replay(
                    reader: asyncio.StreamReader,
                ) -> tuple[dict[str, Any], list[bytes]] | None:
                    nonlocal replay_seen
                    if replay_seen:
                        replay_processed.set()
                    frame = await read_frame(reader)
                    if frame is not None and frame[0].get("type") == "result":
                        replay_seen = True
                    return frame

                monkeypatch.setattr(session_module, "read_frame", observe_replay)
                resumed = RemoteWorker(
                    host,
                    port,
                    TOKEN,
                    registry,
                    name="producer",
                    value_store=store,
                    engine_instance_id=worker.engine_instance_id,
                    resume_from=worker,
                )
                await resumed.start()
                assert resumed._session is session
                await asyncio.wait_for(replay_processed.wait(), 10)
                await asyncio.sleep(0)
            assert not acks, "neither replay nor cancellation may release a live capture"
            assert not run.done()
            assert "produce" in session._invocation_keys
            assert "produce" in session._received_asset_pins
            transfer = session._blob_transfer
            assert transfer is not None
            # Replace the conversation's query pins, leaving only the invocation's hold.
            await transfer.answer_query({"requestId": "pressure", "digests": []})
            store.put(b"another query's cache pressure")
            assert store.has(ref.digest)
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(run, 10)
                assert closing is not None
                await asyncio.wait_for(closing, 10)
            else:
                result = await asyncio.wait_for(run, 10)
                assert result.error is None, result.error
                assert result.outputs is not None
                assert acks == ["resultAck"]
            assert finished.is_set()
            assert not session._received_asset_pins
            assert not session._asset_adoptions
            assert not session._invocation_keys
            await worker.close()
            await stop_service(proc)
            shutil.rmtree(tmp_path / "producer")
            store.put(b"pressure after adoption")
            assert not store.has(ref.digest)
            video = cast(Mapping[str, object], captured["video"].resolve())
            source = cast(AssetRef, video["source"])
            assert digest_bytes(source.read_bytes()) == ref.digest
        finally:
            release.set()
            if run is not None:
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
            if closing is not None:
                await closing
            await worker.close()
            await stop_service(proc)

    run_scenario(scenario())
