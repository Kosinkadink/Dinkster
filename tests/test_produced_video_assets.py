"""VIDEO sources outlive the worker and the transient transfer cache."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetResolver,
    AssetVault,
    digest_bytes,
    register_video_value_type,
)
from dinkster_caches import BudgetedDiskCAS
from dinkster_protocol import Invocation, InvocationResult
from dinkster_values import TypeRegistry, Value, ValueMeta, default_encode, make_list_value
from dinkster_values.model import PyObjPayload
from dinkster_workers import BoundaryDiagnostic, IsolatedWorker, RemoteWorker
from dinkster_workers import session as session_module
from dinkster_workers.blobs import attribute_moved_wire
from dinkster_workers.boundary import ValueCodec, encode_result, read_frame
from dinkster_workers.produced_assets import (
    ProducedAssetAuthority,
    result_asset_digests,
    source_asset_files,
)
from test_remote import TOKEN, start_service, stop_service
from test_reservations import core_registry, make_invocation


@pytest.fixture
def vaults(tmp_path: Path) -> tuple[AssetVault, AssetVault]:
    return AssetVault(tmp_path / "producer"), AssetVault(tmp_path / "engine")


def _publish(vault: AssetVault, data: bytes) -> AssetRef:
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    return AssetRef(digest, "source.mkv", len(data), resolver=vault)


def _value(ref: AssetRef, type_id: str = "comfy.VIDEO") -> Value:
    return Value(
        type_id=type_id,
        fingerprint="source",
        meta=ValueMeta({"asset_refs": [ref.to_wire()]}),
        payload=PyObjPayload({"source": ref, "edits": []}),
    )


@pytest.mark.parametrize(
    "type_id", ["comfy.VIDEO", "dinkster.layers", "comfy.AUDIO", "custom.asset"]
)
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


@pytest.mark.parametrize("recorded", [False, True])
def test_existing_source_requires_integrity_verification(vaults, recorded) -> None:
    _, engine = vaults
    ref = _publish(engine, b"good")

    class Unrecorded:
        def resolve(self, digest: str) -> Path | None:
            return engine.resolve(digest)

    authority = ProducedAssetAuthority(engine, existing=engine if recorded else Unrecorded())
    outputs = {"video": _value(ref)}
    authority.capture(outputs)
    ref.local_path().write_bytes(b"evil")
    with pytest.raises(AssetError, match="integrity failure"):
        authority.capture(outputs)


@pytest.mark.parametrize(
    "type_id", ["comfy.VIDEO", "dinkster.layers", "comfy.AUDIO", "custom.asset"]
)
def test_source_return_does_not_materialize_payloads(
    vaults: tuple[AssetVault, AssetVault], type_id: str
) -> None:
    class Unreadable:
        transport = "pyobj"

        def load(self) -> object:
            raise AssertionError("asset return must not decode payloads")

    producer, engine = vaults
    ref = _publish(producer, b"encoded source")
    value = Value(type_id, "source", ValueMeta({"asset_refs": [ref.to_wire()]}), Unreadable())
    outputs = {"values": make_list_value(type_id, (value, value))}
    assert source_asset_files(outputs, producer) == [(ref.digest, ref.local_path())]
    ProducedAssetAuthority(engine).capture(outputs, producer)
    assert engine.has(ref.digest)


@pytest.mark.parametrize(
    "type_id", ["comfy.VIDEO", "dinkster.layers", "comfy.AUDIO", "custom.asset"]
)
@pytest.mark.parametrize("asset_input", [None, "dinkster.asset", "asset<dinkster.video>"])
def test_returned_input_sources_need_no_produced_asset_vault(
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


@pytest.mark.parametrize(
    "type_id", ["comfy.VIDEO", "dinkster.layers", "comfy.AUDIO", "custom.asset"]
)
def test_source_pins_include_list_children_and_parent_metadata(type_id: str) -> None:
    child = AssetRef(digest_bytes(b"child"), "child", 5)
    parent = AssetRef(digest_bytes(b"parent"), "parent", 6)
    blobs = [
        default_encode({"asset_refs": [child.to_wire()]}),
        default_encode({"asset_refs": [parent.to_wire()]}),
    ]
    header = {
        "outputs": {
            "values": {
                "typeId": "dinkster.list",
                "metaBlob": 1,
                "elements": [{"typeId": type_id, "metaBlob": 0}],
            }
        }
    }
    assert result_asset_digests(header, blobs) == {child.digest, parent.digest}


@pytest.fixture
def producer_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "vproducer"\nnamespaces = ["vproducer"]\n'
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
    register_video_value_type(registry, "comfy.VIDEO", engine)
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

    asyncio.run(scenario())


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
    register_video_value_type(registry, "comfy.VIDEO", engine)
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

    asyncio.run(scenario())


@pytest.mark.parametrize("pressure", [False, True])
@pytest.mark.parametrize("capture_failure", [False, True])
def test_buffered_result_and_eof_retain_source_until_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pressure: bool, capture_failure: bool
) -> None:
    engine = AssetVault(tmp_path / "engine")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(engine.root))
    store = BudgetedDiskCAS(tmp_path / "transfer", max_bytes=1)
    data = b"new producer source"
    ref = AssetRef(store.put(data), "source", len(data))

    async def scenario() -> None:
        registry = core_registry()
        session = session_module.BoundarySession(
            registry, role="test", pack="test", codec=ValueCodec(registry, use_shm=False)
        )
        session.enable_blob_transfer(store)
        closed = False

        class Writer:
            def close(self) -> None:
                nonlocal closed
                closed = True
                assert not session._asset_adoptions
                if pressure:
                    store.put(b"another session publishing during EOF cleanup")

            async def wait_closed(self) -> None:
                pass

        async def drop(*args: object) -> None:
            pass

        def capture(*args: object) -> None:
            raise AssetError("injected capture failure")

        if capture_failure:
            monkeypatch.setattr(session._produced_assets, "capture", capture)
        monkeypatch.setattr(session, "_send_unlocked", drop)
        monkeypatch.setattr(session, "_writer", Writer())
        session._alive = True
        reader = asyncio.StreamReader()
        session._reader = reader
        session._reader_task = asyncio.create_task(session._read_loop())
        pending = asyncio.create_task(session.invoke(make_invocation(0)))
        await asyncio.sleep(0)
        value = replace(
            registry.wrap("core.string", "source descriptor"),
            meta=ValueMeta({"asset_refs": [ref.to_wire()]}),
        )
        header, blobs, _ = encode_result(
            ValueCodec(registry, use_shm=False),
            InvocationResult(outputs={"source": value}),
            "i1",
            0,
        )
        encoded = json.dumps({**header, "blobs": [len(blob) for blob in blobs]}).encode()
        reader.feed_data(len(encoded).to_bytes(4, "big") + encoded + b"".join(blobs))
        reader.feed_eof()
        try:
            result = await asyncio.wait_for(pending, 5)
            assert closed
            if capture_failure:
                assert result.error is not None
                assert result.error.message == "injected capture failure"
                assert not engine.has(ref.digest)
            else:
                assert result.error is None, result.error
                assert replace(ref, resolver=engine).read_bytes() == data
            assert not session._received_asset_pins
            assert not session._asset_adoptions
            store.put(b"pressure after finalization")
            assert not store.has(ref.digest)
        finally:
            await session.close()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())


def test_source_transfer_accounting_deduplicates_across_edges() -> None:
    ref = AssetRef(digest_bytes(b"encoded"), "source", 7)
    blobs = [default_encode({"asset_refs": [ref.to_wire()]})]
    wire = {"typeId": "comfy.VIDEO", "metaBlob": 0, "payload": {"transport": "inline"}}
    header: dict[str, Any] = {
        "outputs": {"first": wire, "second": wire},
        "outputStats": {"first": {"networkBytes": 3}, "second": {"networkBytes": 3}},
    }
    attribute_moved_wire(header, {ref.digest: (7, 1.0)}, blobs)
    assert header["outputStats"]["first"]["networkBytes"] == 10
    assert header["outputStats"]["second"]["networkBytes"] == 3


@pytest.mark.parametrize("index", [-1, True, 1, "0", None])
def test_source_pin_rejects_invalid_metadata_index(index: object) -> None:
    with pytest.raises(AssetError, match="index is invalid"):
        result_asset_digests(
            {"outputs": {"video": {"typeId": "comfy.VIDEO", "metaBlob": index}}},
            [default_encode({})],
        )
