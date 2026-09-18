"""M5 first slice: the boundary crosses machines. A RemoteWorker talks to a
dinkster_workers.service daemon over authenticated TCP and the engine cannot
tell (hazard H3): same graph, same fingerprints (H4), payloads inline or
content-addressed both ways (never shm - the peers do not share memory),
device facts qualified into the remote's namespace, and every failure
conservative - a dead connection fails invocations as NodeErrors and
collapses relay footprints to zero, never claiming bytes were freed. The
daemon outlives its clients: disconnecting is not shutdown."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Coroutine
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import MemoryGovernor
from dinkster_protocol import (
    DeviceResourceId,
    ReplicaId,
    ReplicaRecipeId,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupId,
    WorkUnitId,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import (
    BoundaryDiagnostic,
    RemoteWorker,
    WorkGroupWorkerLane,
    compose_workgroup_configuration,
)
from dinkster_workers.service import READY_LINE_PREFIX
from test_isolated import DEV_MANIFEST, FOUNDATION_MANIFEST, core_registry, image_graph
from test_relay import eventually

TESTS_DIR = Path(__file__).parent
TOKEN = "remote-secret-token-0123456789ab"

TLS_DIR = TESTS_DIR / "tls"
"""Committed test-only certificates (see tests/tls/README.md)."""
TLS_CERT = TLS_DIR / "service-cert.pem"
TLS_KEY = TLS_DIR / "service-key.pem"
TLS_OTHER_CERT = TLS_DIR / "other-cert.pem"
TLS_ARGS = ("--tls-cert", str(TLS_CERT), "--tls-key", str(TLS_KEY))


async def start_service(
    manifest: Path,
    tmp_path: Path,
    *extra_args: str,
    pythonpath: tuple[Path, ...] = (),
    port: int = 0,
) -> tuple[asyncio.subprocess.Process, str, int]:
    """Launch the daemon on an ephemeral port (or ``port``, for a restart
    that must reuse an endpoint) and read the announced endpoint from its
    stdout ready line."""
    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    search_path = os.pathsep.join(str(entry) for entry in (*pythonpath, TESTS_DIR))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dinkster_workers.service",
        "--listen",
        f"127.0.0.1:{port}",
        "--manifest",
        str(manifest),
        "--token-file",
        str(token_file),
        *extra_args,
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": search_path},
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


async def stop_service(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 10.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()


def remote_worker(host: str, port: int, registry: TypeRegistry, **kwargs: object) -> RemoteWorker:
    return RemoteWorker(
        host,
        port,
        TOKEN,
        registry,
        name="box1",
        **kwargs,  # type: ignore[arg-type]
    )


def make_engine(registry: TypeRegistry, worker: RemoteWorker) -> Engine:
    return Engine(
        schemas=dict(worker.schemas),
        registry=registry,
        worker=worker,
        cache=MemoryLRUCache(),
    )


def big_image_graph() -> Graph:
    """Payloads far above DEFAULT_SHM_THRESHOLD: locally these would ride
    shm; remotely they must ride inline and still work."""
    return Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 512, "height": 256}),
            "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
            "s": GraphNode("dev.image.stats", {"image": Link("i", "image")}),
        }
    )


def test_remote_lazy_route_executes_only_the_selected_dynamic_member(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(FOUNDATION_MANIFEST, tmp_path)
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry)
            await worker.start()
            try:
                engine = make_engine(registry, worker)
                graph = Graph(
                    {
                        "first": GraphNode("dinkster.string", {"value": "first"}),
                        "second": GraphNode("dinkster.string", {"value": "second"}),
                        "route": GraphNode(
                            "dinkster.route.switch",
                            {
                                "index": 1,
                                "values.first": Link("first", "value"),
                                "values.second": Link("second", "value"),
                            },
                        ),
                    }
                )

                result = await engine.run(graph, ["route"])
                assert result.outputs["route"]["value"].resolve() == "second"
                assert set(result.executed) == {"second", "route"}
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_service_main_pins_argv_before_loading_the_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The service pins ``sys.argv`` to the explicit ``--comfy-args-json``
    list (default: empty) before the pack loads, exactly like the launched
    worker host: packs that read ComfyUI's import-time CLI must never see
    the service's own arguments."""
    from dinkster_workers import service as service_module

    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    base = (
        "dinkster_workers.service",
        "--listen",
        "127.0.0.1:0",
        "--manifest",
        str(DEV_MANIFEST),
        "--token-file",
        str(token_file),
    )
    pinned: list[list[str]] = []

    def fake_run(coro: Coroutine[object, object, object]) -> None:
        coro.close()
        pinned.append(list(sys.argv))

    monkeypatch.setattr(asyncio, "run", fake_run)
    for extra, expected in (
        ((), []),
        (("--comfy-args-json", '["--fp32-vae", "--cpu"]'), ["--fp32-vae", "--cpu"]),
    ):
        pinned.clear()
        monkeypatch.setattr(sys, "argv", [*base, *extra])
        service_module.main()
        assert pinned == [["dinkster_workers.service", *expected]]


def test_service_main_refuses_malformed_comfy_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    from dinkster_workers import service as service_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.service",
            "--listen",
            "127.0.0.1:0",
            "--manifest",
            str(DEV_MANIFEST),
            "--token-file",
            str(token_file),
            "--comfy-args-json",
            '{"not": "a list"}',
        ],
    )
    with pytest.raises(SystemExit):
        service_module.main()


def test_remote_graph_matches_local_fingerprints_and_never_shm(
    tmp_path: Path,
) -> None:
    """The engine runs the same graph through a remote service and through
    an in-process-launched isolated worker; fingerprints must be identical
    (H4: location independence), and no remote crossing may ever report
    shm - big payloads ride cas (negotiated), small ones inline."""

    async def run_remote() -> tuple[dict[str, str], dict[str, str]]:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        diagnostics: list[BoundaryDiagnostic] = []
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry, on_diagnostic=diagnostics.append)
            await worker.start()
            try:
                engine = make_engine(registry, worker)
                small = await engine.run(image_graph(), ["s", "b"])
                big = await engine.run(big_image_graph(), ["s"])
                assert diagnostics, "remote invocations must report diagnostics"
                for diag in diagnostics:
                    for edge in (*diag.inputs, *diag.outputs):
                        assert edge.transport in ("inline", "cas")
                return (
                    {o: v.fingerprint for o, v in small.outputs["s"].items()},
                    {o: v.fingerprint for o, v in big.outputs["s"].items()},
                )
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    async def run_local() -> tuple[dict[str, str], dict[str, str]]:
        from dinkster_workers import IsolatedWorker

        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry)
        await worker.start()
        try:
            engine = make_engine(registry, worker)  # type: ignore[arg-type]
            small = await engine.run(image_graph(), ["s", "b"])
            big = await engine.run(big_image_graph(), ["s"])
            return (
                {o: v.fingerprint for o, v in small.outputs["s"].items()},
                {o: v.fingerprint for o, v in big.outputs["s"].items()},
            )
        finally:
            await worker.close()

    async def scenario() -> None:
        assert await run_remote() == await run_local()

    asyncio.run(scenario())


def test_remote_big_payloads_ride_cas_and_dedup(tmp_path: Path) -> None:
    """A big value consumed by two invocations crosses the wire once: the
    first crossing negotiates cas and carries bytes, the second is
    digest-only - and if dedup resolution were broken, the second
    invocation would fail instead of blending correctly."""

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        diagnostics: list[BoundaryDiagnostic] = []
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry, on_diagnostic=diagnostics.append)
            await worker.start()
            try:
                engine = make_engine(registry, worker)
                # g's big image feeds both i and b: same payload, two
                # invocations, two crossings - one blob.
                graph = Graph(
                    nodes={
                        "g": GraphNode("dev.image.gradient", {"width": 512, "height": 256}),
                        "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
                        "b": GraphNode(
                            "dev.image.blend",
                            {
                                "a": Link("g", "image"),
                                "b": Link("i", "image"),
                                "ratio": 0.5,
                            },
                        ),
                        "s": GraphNode("dev.image.stats", {"image": Link("b", "image")}),
                    }
                )
                result = await engine.run(graph, ["s"])
                assert result.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
                big_edges = [
                    edge
                    for diag in diagnostics
                    for edge in (*diag.inputs, *diag.outputs)
                    if edge.size_bytes >= 256 * 1024
                ]
                assert big_edges, "the gradient must be big enough to matter"
                assert all(edge.transport == "cas" for edge in big_edges)
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_wrong_token_is_refused(tmp_path: Path) -> None:
    from dinkster_workers import TransportError

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            worker = RemoteWorker(
                host,
                port,
                "x" * len(TOKEN),
                core_registry(),
                name="box1",
                connect_timeout=10.0,
            )
            with pytest.raises((TransportError, RuntimeError), match="handshake"):
                await worker.start()
            await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_stale_protocol_client_is_refused_at_hello(tmp_path: Path) -> None:
    from dinkster_workers import PROTOCOL_VERSION
    from dinkster_workers.boundary import read_frame, write_frame

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            reader, writer = await asyncio.open_connection(host, port)
            try:
                writer.write(TOKEN.encode("utf-8"))
                await writer.drain()
                await write_frame(
                    writer,
                    {
                        "type": "clientHello",
                        "protocol": PROTOCOL_VERSION - 1,
                        "payloadTransports": ["inline", "cas"],
                    },
                    [],
                )
                frame = await asyncio.wait_for(read_frame(reader), timeout=10.0)
                assert frame is not None
                header, _ = frame
                assert header["type"] == "error"
                assert "protocol mismatch" in str(header["message"])
                assert f"service speaks {PROTOCOL_VERSION}" in str(header["message"])
            finally:
                writer.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_ordinary_remote_service_stays_workgroup_capability_absent(tmp_path: Path) -> None:
    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            worker = remote_worker(host, port, core_registry())
            await worker.start()
            try:
                assert worker.instance_token is not None
                assert worker.workgroup_capabilities == frozenset()
                lane = WorkGroupWorkerLane(
                    replica=ReplicaId("replica-a"),
                    worker=WorkerInstanceId(worker.instance_token),
                    device=DeviceResourceId("device-a"),
                    recipe=ReplicaRecipeId("sha256:" + "a" * 64),
                    unit=WorkUnitId("unit-a"),
                    slot=SemanticSlot.SINGLE,
                    boundary=worker,
                )
                with pytest.raises(ValueError, match="do not negotiate"):
                    compose_workgroup_configuration(
                        WorkGroupId("group-a"), WorkGroupAttempt(1), (lane,)
                    )
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_daemon_outlives_clients_and_refuses_concurrent_ones(tmp_path: Path) -> None:
    """One conversation at a time: a second authenticated client is refused
    with the service's own words. Disconnecting the first client ends only
    that conversation - the daemon serves the next engine, warm."""

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            registry = core_registry()
            first = remote_worker(host, port, registry)
            await first.start()
            try:
                second = remote_worker(host, port, core_registry())
                with pytest.raises(RuntimeError, match="busy"):
                    await second.start()
                await second.close()
            finally:
                await first.close()

            # The daemon frees its one-conversation slot when it notices the
            # first client's disconnect - eventually, not instantly. Poll
            # until the slot opens rather than assuming a timing window.
            third = remote_worker(host, port, core_registry())
            deadline = asyncio.get_running_loop().time() + 10.0
            while True:
                try:
                    await third.start()
                    break
                except RuntimeError as exc:
                    if "busy" not in str(exc):
                        raise
                    if asyncio.get_running_loop().time() > deadline:
                        raise
                    await third.close()
                    await asyncio.sleep(0.05)
                    third = remote_worker(host, port, core_registry())
            try:
                engine = make_engine(core_registry(), third)
                result = await engine.run(image_graph(), ["s"])
                assert result.outputs["s"]
            finally:
                await third.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_daemon_waits_for_detached_sync_invocation_before_handover(tmp_path: Path) -> None:
    """A disconnected engine's synchronous node keeps the conversation slot
    until its thread exits. The next engine cannot overlap it, and the warm
    pack's resident memory remains accounted when handover completes."""

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
        )
        entered = tmp_path / "thread-entered"
        finish = tmp_path / "thread-finish"
        vram_remote = "vram:cuda:0@box1"
        ram_remote = "ram@box1"
        governor = MemoryGovernor({vram_remote: 1000, ram_remote: 1000})
        proc, host, port = await start_service(manifest, tmp_path)

        def footprint(device: str) -> int:
            consumers = governor.status().get(device, {}).get("consumers", {})
            return int(consumers.get("memorypack:models", 0))  # type: ignore[union-attr, arg-type]

        try:
            registry = core_registry()
            first = remote_worker(host, port, registry, governor=governor)
            await first.start()
            first_engine = make_engine(registry, first)
            loaded = await first_engine.run(
                Graph(
                    nodes={
                        "load": GraphNode(
                            "mem.load",
                            {"name": "retained", "vram": 400, "ram": 300},
                        )
                    }
                ),
                ["load"],
            )
            assert loaded.outputs["load"]
            assert await eventually(lambda: footprint(vram_remote) == 400)
            assert footprint(ram_remote) == 300

            blocked_run = asyncio.create_task(
                first_engine.run(
                    Graph(
                        nodes={
                            "hold": GraphNode(
                                "mem.thread_hold",
                                {"entered": str(entered), "finish": str(finish)},
                            )
                        }
                    ),
                    ["hold"],
                )
            )
            assert await eventually(entered.exists)
            await first.close()
            assert await eventually(lambda: footprint(vram_remote) == 0)
            assert footprint(ram_remote) == 0

            refusal_deadline = asyncio.get_running_loop().time() + 0.5
            refusals = 0
            while asyncio.get_running_loop().time() < refusal_deadline:
                refused = remote_worker(host, port, core_registry(), governor=governor)
                with pytest.raises(RuntimeError, match="busy"):
                    await refused.start()
                await refused.close()
                refusals += 1
                await asyncio.sleep(0.05)
            assert refusals >= 3
            assert not finish.exists()

            finish.touch()
            with pytest.raises(ExecutionError, match="remote worker 'memorypack' is not running"):
                await blocked_run

            second_registry = core_registry()
            second = remote_worker(host, port, second_registry, governor=governor)
            deadline = asyncio.get_running_loop().time() + 10.0
            while True:
                try:
                    await second.start()
                    break
                except RuntimeError as exc:
                    if "busy" not in str(exc):
                        raise
                    if asyncio.get_running_loop().time() > deadline:
                        raise
                    await second.close()
                    await asyncio.sleep(0.05)
                    second = remote_worker(host, port, second_registry, governor=governor)
            try:
                assert await eventually(lambda: footprint(vram_remote) == 400)
                assert footprint(ram_remote) == 300
                result = await make_engine(second_registry, second).run(
                    Graph(nodes={"state": GraphNode("mem.consumer_state", {})}),
                    ["state"],
                )
                assert result.outputs["state"]["created"].resolve() == 1
                assert result.outputs["state"]["live"].resolve() == 1
            finally:
                await second.close()
        finally:
            finish.touch(exist_ok=True)
            await stop_service(proc)

    asyncio.run(scenario())


def test_service_death_fails_pending_invocations(tmp_path: Path) -> None:
    """Killing the service mid-invocation must resolve the pending call as
    a NodeError naming the worker - never a hang, never a fabricated
    result."""

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "isopack"\n\n[pack.entry]\n'
            'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
        )
        proc, host, port = await start_service(manifest, tmp_path)
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry)
            await worker.start()
            try:
                engine = make_engine(registry, worker)
                graph = Graph(
                    nodes={"sleep": GraphNode("iso.sleepy", {"value": "zzz", "seconds": 30.0})}
                )
                run = asyncio.create_task(engine.run(graph, ["sleep"]))
                await asyncio.sleep(0.5)  # let the invocation reach the service
                proc.kill()
                await proc.wait()
                with pytest.raises(Exception, match="remote worker 'isopack'"):
                    await asyncio.wait_for(run, 30.0)
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_relay_qualifies_devices_and_collapses_on_death(
    tmp_path: Path,
) -> None:
    """The remote's memory consumers register with the local governor under
    remote-qualified residency classes - its cuda:0 is never our cuda:0,
    its ram is never our ram - and the moment the connection dies their
    footprints read zero."""

    async def scenario() -> None:
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "memorypack"\n\n[pack.entry]\n'
            'nodes = "memorypack_nodes:NODES"\ntypes = "memorypack_nodes:register_types"\n'
            'consumers = "memorypack_nodes:memory_consumers"\n'
        )
        vram_remote = "vram:cuda:0@box1"
        ram_remote = "ram@box1"
        governor = MemoryGovernor({vram_remote: 1000, ram_remote: 1000})
        proc, host, port = await start_service(manifest, tmp_path)
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry, governor=governor)
            await worker.start()
            try:
                consumers = governor.status()[vram_remote]["consumers"]
                assert "memorypack:models" in consumers  # type: ignore[operator]

                engine = make_engine(registry, worker)
                await engine.run(
                    Graph(
                        nodes={
                            "load": GraphNode(
                                "mem.load",
                                {"name": "sd15.safetensors", "vram": 400, "ram": 300},
                            )
                        }
                    ),
                    ["load"],
                )

                def footprint(device: str) -> int:
                    consumers = governor.status().get(device, {}).get("consumers", {})
                    return int(consumers.get("memorypack:models", 0))  # type: ignore[union-attr, arg-type]

                assert await eventually(lambda: footprint(vram_remote) == 400)
                assert footprint(ram_remote) == 300
                # Nothing unqualified: the remote's bytes never land on
                # local-namespace lanes.
                assert footprint("vram:cuda:0") == 0
                assert footprint("ram") == 0

                proc.kill()
                await proc.wait()
                assert await eventually(lambda: footprint(vram_remote) == 0)
                assert footprint(ram_remote) == 0
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_remote_never_accepts_shm_descriptors() -> None:
    """Defense in depth: even if a peer advertised an shm payload, the
    remote codec must refuse it - a segment name from another machine
    would attach to unrelated local memory."""
    from dinkster_workers import BoundaryError, ValueCodec

    registry = TypeRegistry()
    register_core_types(registry)
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    wire = {
        "typeId": "core.string",
        "fingerprint": "f",
        "metaBlob": 0,
        "payload": {"transport": "shm", "segment": "dinksterdeadbeef", "size": 4},
    }
    from dinkster_values import default_encode

    with pytest.raises(BoundaryError, match="shm payloads are not accepted"):
        codec.decode(wire, [default_encode({})], [])


def test_tls_round_trip_executes(tmp_path: Path) -> None:
    """A daemon serving with --tls-cert/--tls-key and a client pinning that
    certificate complete the handshake and execute a graph exactly like the
    plaintext pair; the wire protocol is untouched by the wrapping."""

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, *TLS_ARGS)
        try:
            registry = core_registry()
            worker = remote_worker(host, port, registry, tls_ca_file=TLS_CERT)
            await worker.start()
            try:
                engine = make_engine(registry, worker)
                result = await engine.run(image_graph(), ["s", "b"])
                assert result.outputs["s"], "TLS round trip must produce outputs"
            finally:
                await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_tls_wrong_ca_is_refused_and_daemon_serves_the_next_client(tmp_path: Path) -> None:
    """A client pinning an unrelated certificate is refused with a message
    naming the verification failure, and the failed handshake leaves the
    daemon free for a correctly configured client."""
    from dinkster_workers import TransportError

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, *TLS_ARGS)
        try:
            registry = core_registry()
            wrong = remote_worker(host, port, registry, tls_ca_file=TLS_OTHER_CERT)
            with pytest.raises(TransportError, match="certificate verification"):
                await wrong.start()
            await wrong.close()

            good = remote_worker(host, port, registry, tls_ca_file=TLS_CERT)
            await good.start()
            await good.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_plaintext_client_against_tls_daemon_fails_fast(tmp_path: Path) -> None:
    """A client without tls_ca_file dialing a TLS daemon gets a prompt
    TransportError that names the TLS possibility, never a hang: the
    daemon's handshake rejects the plaintext bytes and closes."""
    from dinkster_workers import TransportError

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, *TLS_ARGS)
        try:
            worker = remote_worker(host, port, core_registry(), connect_timeout=10.0)
            with pytest.raises(TransportError, match="TLS daemon reached without tls_ca_file"):
                await worker.start()
            await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_tls_client_against_plaintext_daemon_fails_fast(tmp_path: Path) -> None:
    """A client with tls_ca_file dialing a plaintext daemon fails the
    handshake promptly with a message naming the mismatch: the daemon reads
    the TLS ClientHello as a wrong token and hangs up."""
    from dinkster_workers import TransportError

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path)
        try:
            worker = remote_worker(
                host, port, core_registry(), tls_ca_file=TLS_CERT, connect_timeout=10.0
            )
            with pytest.raises(TransportError, match="TLS was requested"):
                await worker.start()
            await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_wrong_token_over_tls_is_refused_silently(tmp_path: Path) -> None:
    """TLS wraps the stream; token auth is unchanged inside it. A bad token
    still gets the silent hangup, never a distinguishable refusal."""
    from dinkster_workers import TransportError

    async def scenario() -> None:
        proc, host, port = await start_service(DEV_MANIFEST, tmp_path, *TLS_ARGS)
        try:
            worker = RemoteWorker(
                host,
                port,
                "x" * len(TOKEN),
                core_registry(),
                name="box1",
                tls_ca_file=TLS_CERT,
                connect_timeout=10.0,
            )
            with pytest.raises(TransportError, match="wrong token"):
                await worker.start()
            await worker.close()
        finally:
            await stop_service(proc)

    asyncio.run(scenario())


def test_service_main_requires_cert_and_key_together(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    from dinkster_workers import service as service_module

    base = [
        "dinkster_workers.service",
        "--listen",
        "127.0.0.1:0",
        "--manifest",
        str(DEV_MANIFEST),
        "--token-file",
        str(token_file),
    ]
    for lonely in (["--tls-cert", str(TLS_CERT)], ["--tls-key", str(TLS_KEY)]):
        monkeypatch.setattr(sys, "argv", [*base, *lonely])
        with pytest.raises(SystemExit):
            service_module.main()


def test_service_refuses_unreadable_tls_material(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "remote-token.txt"
    token_file.write_text(TOKEN, encoding="utf-8")
    from dinkster_workers import service as service_module
    from dinkster_workers.service import ServiceError

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.service",
            "--listen",
            "127.0.0.1:0",
            "--manifest",
            str(DEV_MANIFEST),
            "--token-file",
            str(token_file),
            "--tls-cert",
            str(tmp_path / "absent-cert.pem"),
            "--tls-key",
            str(tmp_path / "absent-key.pem"),
        ],
    )
    with pytest.raises(ServiceError, match="cannot load TLS certificate"):
        service_module.main()
