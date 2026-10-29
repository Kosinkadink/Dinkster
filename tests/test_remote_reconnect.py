"""Remote worker reconnect: a composed remote whose session died is
redialed and swapped back in atomically (reattach_remote) - adopting a
redeployed daemon's changed surface, refusing a change that would strand
dependents, and telling a daemon restart (cache-clearing) from a
transport-only reconnect - while the serve-side supervisor drives redials
from liveness polling with bounded backoff."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, sentinel

import pytest
from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import BudgetedDiskCAS
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_server import STATE_KEY, create_app
from dinkster_workers import TransportError
from dinkster_workers.session import WorkerDied
from test_isolated import write_iso_manifest
from test_remote import start_service, stop_service
from test_serve_remote import (
    composition_snapshot,
    daemon_accepts_a_client,
    remote_spec,
    wait_for_job,
    write_composed_iso_manifest,
)

from dinkster.compose import CompositionError, PackDelta, ServingComposer, UnknownPackError
from dinkster.remote_reconnect import (
    ReconnectPolicy,
    RemoteReconnectSupervisor,
    announce_remote_delta,
    backoff_delays,
)
from dinkster.remotes import RemoteSpec

TESTS_DIR = Path(__file__).parent


def write_variant_manifest(root: Path, dropped: str) -> Path:
    """A daemon manifest announcing the isopack surface minus one node
    type, through a generated module beside the manifest (add ``root`` to
    the daemon's pythonpath)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "isopack_nodes_variant.py").write_text(
        "from isopack_nodes import NODES as _ALL, register_types as register_types\n"
        f"NODES = [node for node in _ALL if node.define_schema().node_type != {dropped!r}]\n",
        encoding="utf-8",
    )
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\n\n[pack.entry]\n'
        'nodes = "isopack_nodes_variant:NODES"\ntypes = "isopack_nodes_variant:register_types"\n'
    )
    return manifest


async def wait_for_connected(composer: ServingComposer, name: str, value: bool) -> None:
    async with asyncio.timeout(10):
        while composer.remote_connected(name) is not value:
            await asyncio.sleep(0.02)


def worker_status(composer: ServingComposer, spec: RemoteSpec) -> str:
    (entry,) = [info for info in composer.workers((spec,)) if info.name != "local"]
    return entry.status


def test_backoff_delays_double_to_the_cap_with_bounded_jitter() -> None:
    policy = ReconnectPolicy(initial_delay=1.0, max_delay=8.0, jitter=0.5)
    stretched = backoff_delays(policy, rand=lambda: 1.0)
    assert [next(stretched) for _ in range(5)] == [1.5, 3.0, 6.0, 12.0, 12.0]
    flat = backoff_delays(policy, rand=lambda: 0.0)
    assert [next(flat) for _ in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_first_remote_redial_prepares_validation_before_publication() -> None:
    state = Mock()
    state.prepare_replace = AsyncMock(return_value=sentinel.validation)
    state.replace.return_value = 2
    delta = PackDelta("box1", {}, {}, {})

    assert asyncio.run(announce_remote_delta(state, delta)) == 2
    state.prepare_replace.assert_awaited_once_with((), (), {}, {}, {})
    assert state.replace.call_args.kwargs["_validation"] is sentinel.validation


def test_reattach_after_daemon_restart_swaps_in_a_new_session(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        restarted = None
        try:
            spec = remote_spec(host, port, tmp_path)
            with pytest.raises(UnknownPackError, match="no composed remote"):
                await composer.reattach_remote(spec)
            await composer.add_remote(spec)
            first_token = composer._remotes["box1"].instance_token
            assert first_token is not None
            await stop_service(proc)
            await wait_for_connected(composer, "box1", False)
            assert worker_status(composer, spec) == "disconnected"
            # A failed redial (daemon still down) changes nothing.
            before = composition_snapshot(composer)
            with pytest.raises(TransportError, match="could not connect"):
                await composer.reattach_remote(spec)
            assert composition_snapshot(composer) == before
            assert worker_status(composer, spec) == "disconnected"

            restarted, _, _ = await start_service(write_iso_manifest(tmp_path), tmp_path, port=port)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is False
            assert outcome.result.pack == "box1"
            assert "iso.chatty" in outcome.result.removed_types
            assert "iso.chatty" in outcome.result.delta.schemas
            assert composer.remote_connected("box1") is True
            assert worker_status(composer, spec) == "connected"
            assert composer._remotes["box1"].instance_token not in (None, first_token)
            assert composer._routing.has_route("iso.chatty")
            # Exactly one live remote connection survives the swap.
            assert len(composer.composition._isolated) == 1
        finally:
            await composer.close()
            if restarted is not None:
                await stop_service(restarted)
            await stop_service(proc)

    asyncio.run(scenario())


def test_reattach_to_the_same_daemon_process_is_same_instance(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            first_token = composer._remotes["box1"].instance_token
            # Engine-side session teardown: the daemon outlives its
            # clients and frees its conversation slot.
            await composer._remotes["box1"].worker.close()
            await wait_for_connected(composer, "box1", False)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            assert composer.remote_connected("box1") is True
            assert composer._remotes["box1"].instance_token == first_token
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_inflight_invocation_resumes_after_same_daemon_network_cut(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "slow": GraphNode(
                        "iso.sleepy",
                        {"value": "survived", "seconds": 1.0},
                    )
                }
            )
            run = asyncio.create_task(engine.run(graph, ["slow"], run_id="resume-job"))
            async with asyncio.timeout(10):
                session = composer._remotes["box1"].worker._session
                while not session._accepted_invocations:
                    await asyncio.sleep(0)
                (accepted,) = session._accepted_invocations.values()
                await accepted.wait()
            writer = composer._remotes["box1"].worker._session._writer
            assert writer is not None
            writer.transport.abort()
            await wait_for_connected(composer, "box1", False)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            result = await asyncio.wait_for(run, 10.0)
            assert result.outputs["slow"]["value"].resolve() == "survived"
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_failed_resume_publication_preserves_invocation_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "slow": GraphNode(
                        "iso.sleepy",
                        {"value": "retried", "seconds": 1.0},
                    )
                }
            )
            run = asyncio.create_task(engine.run(graph, ["slow"], run_id="retry-job"))
            session = composer._remotes["box1"].worker._session
            async with asyncio.timeout(10):
                while not session._accepted_invocations:
                    await asyncio.sleep(0)
                (accepted,) = session._accepted_invocations.values()
                await accepted.wait()
            assert session._writer is not None
            session._writer.transport.abort()
            await wait_for_connected(composer, "box1", False)

            validate = composer._validated_remote_surface

            def fail_validation(*args: object, **kwargs: object) -> object:
                del args, kwargs
                raise CompositionError("injected publication failure")

            monkeypatch.setattr(composer, "_validated_remote_surface", fail_validation)
            with pytest.raises(CompositionError, match="injected"):
                await composer.reattach_remote(spec)
            assert not run.done()
            assert session._invocation_keys

            monkeypatch.setattr(composer, "_validated_remote_surface", validate)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            result = await asyncio.wait_for(run, 10.0)
            assert result.outputs["slow"]["value"].resolve() == "retried"
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_completed_result_replays_after_reconnect(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "slow": GraphNode(
                        "iso.sleepy",
                        {"value": "replayed", "seconds": 0.2},
                    )
                }
            )
            run = asyncio.create_task(engine.run(graph, ["slow"], run_id="replay-job"))
            session = composer._remotes["box1"].worker._session
            async with asyncio.timeout(10):
                while not session._accepted_invocations:
                    await asyncio.sleep(0)
                (accepted,) = session._accepted_invocations.values()
                await accepted.wait()
            assert session._writer is not None
            session._writer.transport.abort()
            await wait_for_connected(composer, "box1", False)
            await asyncio.sleep(0.4)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            result = await asyncio.wait_for(run, 10.0)
            assert result.outputs["slow"]["value"].resolve() == "replayed"
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_result_replays_when_acknowledgement_transport_is_lost(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            session = composer._remotes["box1"].worker._session
            send = session.send
            cut = asyncio.Event()

            async def cut_first_ack(
                header: dict[str, object],
                blobs: Sequence[bytes],
                segments: Sequence[SharedMemory] = (),
            ) -> None:
                if header.get("type") == "resultAck" and not cut.is_set():
                    assert session._writer is not None
                    session._writer.transport.abort()
                    cut.set()
                    raise WorkerDied()
                await send(header, blobs, segments)

            session.send = cut_first_ack  # type: ignore[method-assign]
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "slow": GraphNode(
                        "iso.sleepy",
                        {"value": "acknowledged", "seconds": 0.1},
                    )
                }
            )
            run = asyncio.create_task(engine.run(graph, ["slow"], run_id="ack-job"))
            await asyncio.wait_for(cut.wait(), 10.0)
            await wait_for_connected(composer, "box1", False)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            result = await asyncio.wait_for(run, 10.0)
            assert result.outputs["slow"]["value"].resolve() == "acknowledged"
            assert not session._invocation_keys
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_persistent_result_transfer_restarts_after_network_cut(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(
            write_iso_manifest(tmp_path),
            tmp_path,
            "--value-store",
            str(tmp_path / "daemon-store"),
        )
        composer = ServingComposer(remote_value_store=BudgetedDiskCAS(tmp_path / "engine-store"))
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            session = composer._remotes["box1"].worker._session
            transfer = session._blob_transfer
            assert transfer is not None
            answer_query = transfer.answer_query
            cut = asyncio.Event()

            async def cut_before_answer(header: dict[str, Any]) -> None:
                if not cut.is_set():
                    assert session._writer is not None
                    session._writer.transport.abort()
                    cut.set()
                await answer_query(header)

            transfer.answer_query = cut_before_answer  # type: ignore[method-assign]
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "out": GraphNode("iso.blob_out", {"size": 512 * 1024}),
                    "len": GraphNode("iso.blob_len", {"blob": Link("out", "blob")}),
                }
            )
            run = asyncio.create_task(engine.run(graph, ["len"], run_id="blob-job"))
            await asyncio.wait_for(cut.wait(), 10.0)
            await wait_for_connected(composer, "box1", False)
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            result = await asyncio.wait_for(run, 20.0)
            assert result.outputs["len"]["length"].resolve() == 512 * 1024
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_cancellation_during_disconnect_is_applied_before_rebind(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "slow": GraphNode(
                        "iso.sleepy",
                        {"value": "cancelled", "seconds": 30.0},
                    )
                }
            )
            run = asyncio.create_task(engine.run(graph, ["slow"], run_id="cancel-job"))
            session = composer._remotes["box1"].worker._session
            async with asyncio.timeout(10):
                while not session._accepted_invocations:
                    await asyncio.sleep(0)
                (accepted,) = session._accepted_invocations.values()
                await accepted.wait()
            assert session._writer is not None
            session._writer.transport.abort()
            await wait_for_connected(composer, "box1", False)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is True
            assert not session._invocation_keys
            quick = Graph(nodes={"quick": GraphNode("iso.sleepy", {"value": "ok", "seconds": 0.0})})
            result = await engine.run(quick, ["quick"])
            assert result.outputs["quick"]["value"].resolve() == "ok"
        finally:
            await composer.close()
            await stop_service(proc)

    asyncio.run(scenario())


def test_reattach_adopts_a_changed_daemon_surface(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        restarted = None
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            assert "iso.chatty" in composer.composition.schemas
            await stop_service(proc)
            await wait_for_connected(composer, "box1", False)

            variant_root = tmp_path / "variant"
            manifest = write_variant_manifest(variant_root, "iso.chatty")
            restarted, _, _ = await start_service(
                manifest, tmp_path, pythonpath=(variant_root,), port=port
            )
            outcome = await composer.reattach_remote(spec)
            assert "iso.chatty" in outcome.result.removed_types
            assert "iso.chatty" not in outcome.result.delta.schemas
            assert "iso.chatty" not in composer.composition.schemas
            assert "iso.chatty" not in composer._remotes["box1"].node_types
            assert not composer._routing.has_route("iso.chatty")
            assert "iso.sleepy" in composer.composition.schemas
            assert composer._routing.has_route("iso.sleepy")
        finally:
            await composer.close()
            if restarted is not None:
                await stop_service(restarted)
            await stop_service(proc)

    asyncio.run(scenario())


def test_reattach_refuses_a_surface_change_that_strands_dependents(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        owner_root.mkdir()
        arm_root = tmp_path / "arm"
        arm_root.mkdir()
        proc1, host1, port1 = await start_service(write_iso_manifest(owner_root), owner_root)
        proc2, host2, port2 = await start_service(write_iso_manifest(arm_root), arm_root)
        composer = ServingComposer()
        restarted = None
        try:
            spec1 = remote_spec(host1, port1, owner_root)
            spec2 = remote_spec(host2, port2, arm_root, name="box2")
            await composer.add_remote(spec1)
            await composer.add_remote(spec2)
            assert composer._owners["iso.chatty"] == "box1"
            assert "iso.chatty" in composer._remotes["box2"].node_types

            await stop_service(proc1)
            await wait_for_connected(composer, "box1", False)
            variant_root = tmp_path / "variant"
            manifest = write_variant_manifest(variant_root, "iso.chatty")
            restarted, _, _ = await start_service(
                manifest, owner_root, pythonpath=(variant_root,), port=port1
            )
            before = composition_snapshot(composer)
            with pytest.raises(CompositionError, match="cannot reattach.*box2.*iso.chatty"):
                await composer.reattach_remote(spec1)
            assert composition_snapshot(composer) == before
            assert composer.remote_connected("box1") is False
            assert composer.remote_connected("box2") is True
            # The refused dial released the daemon's conversation slot.
            await daemon_accepts_a_client(host1, port1)
        finally:
            await composer.close()
            if restarted is not None:
                await stop_service(restarted)
            await stop_service(proc2)
            await stop_service(proc1)

    asyncio.run(scenario())


def test_reattach_aborts_when_the_old_session_cannot_be_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old session is reaped before the commit point: a failing reap
    aborts the whole swap (composition untouched, fresh dial released)
    rather than leaving a half-committed composer."""

    async def scenario() -> None:
        proc, host, port = await start_service(write_iso_manifest(tmp_path), tmp_path)
        composer = ServingComposer()
        restarted = None
        try:
            spec = remote_spec(host, port, tmp_path)
            await composer.add_remote(spec)
            await stop_service(proc)
            await wait_for_connected(composer, "box1", False)
            restarted, _, _ = await start_service(write_iso_manifest(tmp_path), tmp_path, port=port)

            async def refuse_close() -> None:
                raise RuntimeError("old session close refused")

            before = composition_snapshot(composer)
            with monkeypatch.context() as patch:
                patch.setattr(composer._remotes["box1"].worker, "close", refuse_close)
                with pytest.raises(RuntimeError, match="old session close refused"):
                    await composer.reattach_remote(spec)
            assert composition_snapshot(composer) == before
            assert worker_status(composer, spec) == "disconnected"
            # The aborted swap released the fresh dial: the daemon's
            # single conversation slot is free again.
            await daemon_accepts_a_client(host, port)
            # With the old session reapable again the same redial lands.
            outcome = await composer.reattach_remote(spec)
            assert outcome.same_instance is False
            assert composer.remote_connected("box1") is True
        finally:
            await composer.close()
            if restarted is not None:
                await stop_service(restarted)
            await stop_service(proc)

    asyncio.run(scenario())


def test_supervisor_redials_and_hinted_jobs_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        local_manifest = write_composed_iso_manifest(tmp_path / "local", "local-iso")
        remote_root = tmp_path / "remote"
        remote_manifest = write_composed_iso_manifest(remote_root, "remote-iso")
        proc, host, port = await start_service(remote_manifest, remote_root)
        composer = ServingComposer(worker_env={"PYTHONPATH": str(TESTS_DIR)})
        client: TestClient | None = None
        supervisor_task: asyncio.Task[None] | None = None
        restarted = None
        try:
            await composer.add_pack(local_manifest)
            spec = remote_spec(host, port, remote_root)
            await composer.add_remote(spec)
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                execution_arms=composition.execution_arms,
                choices=composition.choices,
                compat_skips=composition.compat_skips,
                workers=lambda: composer.workers((spec,)),
                place_execution=composer.place_execution,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            state = app[STATE_KEY]
            (key,) = state.seed_composition(["box1"])
            state.mark_pack_announced(key, 1)
            prepare_replace = AsyncMock(wraps=state.prepare_replace)
            monkeypatch.setattr(state, "prepare_replace", prepare_replace)
            supervisor = RemoteReconnectSupervisor(
                state,
                composer,
                (spec,),
                (key,),
                policy=ReconnectPolicy(
                    initial_delay=0.05, max_delay=0.2, jitter=0.0, poll_interval=0.02
                ),
            )
            supervisor_task = asyncio.create_task(supervisor.run())

            async def hinted_job(client_id: str) -> ClientResponse:
                graph = Graph(nodes={"n": GraphNode("iso.chatty", {"value": client_id})})
                return await client.post(
                    "/api/jobs",
                    json={
                        "clientId": client_id,
                        "jobId": "one",
                        "graph": graph_to_wire(graph),
                        "targets": ["n"],
                        "placement": {"n": "box1"},
                    },
                )

            assert (await hinted_job("warm")).status == 202
            assert (await wait_for_job(client, "warm", "one"))["state"] == "completed"

            await stop_service(proc)
            await wait_for_connected(composer, "box1", False)
            # While down: submission hints naming box1 are refused and the
            # composition report row flips to failed.
            assert (await hinted_job("down")).status == 400
            async with asyncio.timeout(10):
                while state.composition_packs[key].get("state") != "failed":
                    await asyncio.sleep(0.02)

            restarted, _, _ = await start_service(remote_manifest, remote_root, port=port)
            async with asyncio.timeout(30):
                while state.composition_packs[key].get("state") != "announced":
                    await asyncio.sleep(0.02)
            prepare_replace.assert_awaited_once()
            assert composer.remote_connected("box1") is True
            assert (await hinted_job("back")).status == 202
            assert (await wait_for_job(client, "back", "one"))["state"] == "completed"
            job = state.queue.get("back", "one")
            assert job is not None
            assert job.node_receipts["n"]["worker"] == "box1"
        finally:
            if supervisor_task is not None:
                supervisor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await supervisor_task
            if client is not None:
                await client.close()
            await composer.close()
            if restarted is not None:
                await stop_service(restarted)
            await stop_service(proc)

    asyncio.run(scenario())
