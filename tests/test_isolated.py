"""M2: the boundary proves itself. An isolated worker process hosts a pack;
the engine cannot tell (hazard H3), fingerprints are location-independent
(H4), the parent never imports pack code (H1/H5), and the crossing's cost
is visible as diagnostics (DESIGN 3.9)."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, ExecutionError, GraphValidationError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_dev import PACK_NODES, register_dev_types
from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    DeviceResourceId,
    ExportSnapshot,
    LazyStatusInvocation,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaBinding,
    ReplicaId,
    ReplicaReady,
    ReplicaRecipeId,
    SemanticSlot,
    WorkerInstanceId,
    WorkGroupAttempt,
    WorkGroupDefinition,
    WorkGroupId,
    WorkGroupReleased,
    WorkUnitDefinition,
    WorkUnitId,
    derive_attention_route_token,
)
from dinkster_schema import (
    build_node_types,
    build_schemas,
    schema_signature,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import (
    BoundaryDiagnostic,
    GroupIsolatedWorker,
    InProcessWorker,
    IsolatedWorker,
    ManifestError,
    RoutingWorker,
    ensure_pack_venv,
    load_manifest,
)
from dinkster_workers import host as host_module
from dinkster_workers.boundary import (
    PROTOCOL_VERSION,
    ValueCodec,
)
from dinkster_workers.doctor import diagnose
from dinkster_workers.launch import HOST_OWNED_ENVIRONMENT, Launcher, LaunchSpec
from dinkster_workers.session import BoundarySession

from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip

REPO_ROOT = Path(__file__).parents[1]
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SIGSTOP = getattr(signal, "SIGSTOP", signal.SIGTERM)
_KILLED_RETURN_CODE = 1 if os.name == "nt" else -_SIGKILL
# Boundary tests exercise the numpy image family, which lives in the DEV
# pack (test scaffolding, never a user surface) - so the isolated worker
# under test hosts dinkster-nodes-dev.
DEV_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"
TESTS_DIR = Path(__file__).parent


def core_registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    return registry


def attention_token(*, torch_version: str = "2.13.0") -> AttentionRouteToken:
    return derive_attention_route_token(
        attention_capabilities(torch_version=torch_version), AttentionPolicyConfig()
    )


def attention_capabilities(*, torch_version: str = "2.13.0") -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime=torch_version,
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa",),
        provider_versions=(("torch", torch_version),),
    )


def write_iso_manifest(
    tmp_path: Path,
    *,
    name: str = "isopack",
    module_name: str = "isopack_nodes",
) -> Path:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
        'type-namespaces = ["iso"]\n\n[pack.entry]\n'
        f'nodes = "{module_name}:NODES"\ntypes = "{module_name}:register_types"\n'
    )
    return manifest


def write_workgroup_manifest(tmp_path: Path, *, factory: str = "workgroup_handler_factory") -> Path:
    manifest = write_iso_manifest(tmp_path)
    with manifest.open("a") as handle:
        handle.write(f'workgroup_handler = "isopack_nodes:{factory}"\n')
    return manifest


def iso_worker(tmp_path: Path, registry: TypeRegistry, **kwargs: object) -> IsolatedWorker:
    return IsolatedWorker(
        write_iso_manifest(tmp_path),
        registry,
        extra_env={"PYTHONPATH": str(TESTS_DIR)},
        **kwargs,  # type: ignore[arg-type]
    )


class DifferentWorkingDirectoryLauncher(Launcher):
    def __init__(self, working_directory: Path, manifests: tuple[Path, ...]) -> None:
        self._working_directory = working_directory
        self._manifests = tuple(path.resolve() for path in manifests)

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        assert spec.pack_root == self._manifests[0].parent
        manifest_arguments = tuple(
            Path(spec.command[index + 1])
            for index, value in enumerate(spec.command)
            if value == "--manifest"
        )
        assert manifest_arguments == self._manifests
        environment = {
            name: value for name, value in os.environ.items() if name not in HOST_OWNED_ENVIRONMENT
        }
        environment.update(spec.env)
        return await asyncio.create_subprocess_exec(
            *spec.command,
            env=environment,
            cwd=self._working_directory,
        )


def test_isolated_worker_imports_pack_from_a_different_working_directory(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    manifest = write_iso_manifest(pack_root)
    shutil.copy(TESTS_DIR / "isopack_nodes.py", pack_root / "isopack_nodes.py")
    working_directory = tmp_path / "working"
    working_directory.mkdir()

    async def scenario() -> None:
        worker = IsolatedWorker(
            manifest.resolve(),
            core_registry(),
            launcher=DifferentWorkingDirectoryLauncher(working_directory, (manifest,)),
        )
        await worker.start()
        try:
            assert "iso.sleepy" in worker.schemas
        finally:
            await worker.close()

    run_scenario(scenario())


def test_group_isolated_worker_imports_packs_from_a_different_working_directory(
    tmp_path: Path,
) -> None:
    manifests: list[Path] = []
    for name in ("first", "second"):
        pack_root = tmp_path / name
        pack_root.mkdir()
        module_name = f"{name}_nodes"
        manifest = write_iso_manifest(pack_root, name=name, module_name=module_name)
        shutil.copy(TESTS_DIR / "isopack_nodes.py", pack_root / f"{module_name}.py")
        manifests.append(manifest)
    working_directory = tmp_path / "working"
    working_directory.mkdir()

    async def scenario() -> None:
        group = GroupIsolatedWorker(
            "cwd-test",
            manifests,
            core_registry(),
            launcher=DifferentWorkingDirectoryLauncher(working_directory, tuple(manifests)),
        )
        await group.start()
        try:
            assert "iso.sleepy" in group.members["first"].schemas
            assert "iso.sleepy" in group.members["second"].schemas
        finally:
            await group.close()

    run_scenario(scenario())


def test_worker_and_doctor_reject_implicit_distribution_source_root(tmp_path: Path) -> None:
    distribution = tmp_path / "distribution"
    pack_root = distribution / "pack"
    pack_root.mkdir(parents=True)
    manifest = write_iso_manifest(pack_root)
    (distribution / "pyproject.toml").write_text("[project]\nname = 'isopack'\nversion = '1'\n")
    source_root = distribution / "src"
    source_root.mkdir()
    shutil.copy(TESTS_DIR / "isopack_nodes.py", source_root / "isopack_nodes.py")

    report = diagnose(manifest)
    assert not report.ok
    assert any(finding.code == "entry.unresolvable" for finding in report.findings)

    async def scenario() -> None:
        worker = IsolatedWorker(manifest, core_registry())
        with pytest.raises(RuntimeError, match="failed to start: exited with code 1"):
            await worker.start()
        await worker.close()

    run_scenario(scenario())


def test_launched_child_resolves_workgroup_handler_without_parent_import(tmp_path: Path) -> None:
    async def scenario() -> None:
        sys.modules.pop("isopack_nodes", None)
        worker = IsolatedWorker(
            write_workgroup_manifest(tmp_path),
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        await worker.start()
        try:
            assert "isopack_nodes" not in sys.modules
            assert worker.instance_token is not None
            replica = ReplicaId("replica-a")
            fields = {
                "worker": WorkerInstanceId(worker.instance_token),
                "replica": replica,
                "group": WorkGroupId("group-a"),
                "attempt": WorkGroupAttempt(1),
                "device": DeviceResourceId("device-a"),
            }
            definition = WorkGroupDefinition(
                fields["group"],
                fields["attempt"],
                (
                    ReplicaBinding(
                        replica,
                        fields["worker"],
                        fields["device"],
                        ReplicaRecipeId("sha256:" + "a" * 64),
                    ),
                ),
                (WorkUnitDefinition(WorkUnitId("unit-a"), replica, SemanticSlot.SINGLE),),
            )
            endpoint = worker.bind_workgroup_endpoint(definition, replica)
            await endpoint.send(
                PrepareReplica(**fields, recipe=ReplicaRecipeId("sha256:" + "a" * 64))
            )
            assert type(await endpoint.receive()) is ReplicaReady
            await endpoint.send(ReleaseWorkGroup(**fields))
            assert type(await endpoint.receive()) is WorkGroupReleased
        finally:
            await worker.close()
        assert "isopack_nodes" not in sys.modules

    run_scenario(scenario())


def test_group_launched_child_keeps_handler_present_and_absent_members_distinct(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        active_dir = tmp_path / "active"
        plain_dir = tmp_path / "plain"
        active_dir.mkdir()
        plain_dir.mkdir()
        active_manifest = write_workgroup_manifest(active_dir)
        plain_manifest = write_iso_manifest(plain_dir, name="plainpack")
        group = GroupIsolatedWorker(
            "workgroup-test",
            (active_manifest, plain_manifest),
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        await group.start()
        try:
            active = group.members["isopack"]
            plain = group.members["plainpack"]
            assert active.workgroup_capabilities == frozenset({WORKGROUP_CAPABILITY})
            assert plain.workgroup_capabilities == frozenset()
            assert active.instance_token is not None
            replica = ReplicaId("replica-a")
            worker_id = WorkerInstanceId(active.instance_token)
            group_id = WorkGroupId("group-a")
            attempt = WorkGroupAttempt(1)
            device = DeviceResourceId("device-a")
            definition = WorkGroupDefinition(
                group_id,
                attempt,
                (
                    ReplicaBinding(
                        replica,
                        worker_id,
                        device,
                        ReplicaRecipeId("sha256:" + "a" * 64),
                    ),
                ),
                (WorkUnitDefinition(WorkUnitId("unit-a"), replica, SemanticSlot.SINGLE),),
            )
            endpoint = active.bind_workgroup_endpoint(definition, replica)
            await endpoint.send(
                PrepareReplica(
                    worker_id,
                    replica,
                    group_id,
                    attempt,
                    device,
                    ReplicaRecipeId("sha256:" + "a" * 64),
                )
            )
            assert type(await endpoint.receive()) is ReplicaReady
        finally:
            await group.close()

    run_scenario(scenario())


@pytest.mark.parametrize("worker_protocol", (PROTOCOL_VERSION, PROTOCOL_VERSION + 1))
def test_separately_shipped_worker_negotiates_protocol(
    monkeypatch: pytest.MonkeyPatch, worker_protocol: int
) -> None:
    monkeypatch.setattr(host_module, "PROTOCOL_VERSION", worker_protocol)

    async def scenario() -> None:
        registry = core_registry()
        accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.set_result(task)
            await host_module.serve_connection(
                reader,
                writer,
                pack_name="own-environment",
                worker=InProcessWorker({}, registry),
                schemas={},
                planner=None,
                consumers={},
                codec=ValueCodec(registry),
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        host_task = await accepted
        session = BoundarySession(
            registry,
            role="isolated worker",
            pack="own-environment",
            codec=ValueCodec(registry),
        )
        try:
            if worker_protocol == PROTOCOL_VERSION:
                await session.begin(reader, writer, timeout=2)
                assert session.alive
                await session.send({"type": "shutdown"}, [])
            else:
                with pytest.raises(RuntimeError, match="protocol mismatch"):
                    await session.begin(reader, writer, timeout=2)
                assert not session.alive
        finally:
            await session.close()
            await asyncio.wait_for(host_task, 2)
            server.close()
            await server.wait_closed()

    run_scenario(scenario())


def image_graph(ratio: float = 0.5) -> Graph:
    return Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 16, "height": 8}),
            "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
            "b": GraphNode(
                "dev.image.blend",
                {"a": Link("g", "image"), "b": Link("i", "image"), "ratio": ratio},
            ),
            "s": GraphNode("dev.image.stats", {"image": Link("b", "image")}),
        }
    )


def test_hello_announces_schemas_in_wire_format() -> None:
    async def scenario() -> None:
        worker = IsolatedWorker(DEV_MANIFEST, core_registry())
        await worker.start()
        try:
            assert worker.pack == "dinkster-nodes-dev"
            assert worker.attention_route_token is None
            local = build_schemas(PACK_NODES)
            assert set(worker.schemas) == set(local)
            # The wire round trip preserves schema identity: signatures match,
            # so cache keys agree across the boundary (hazard H4).
            for node_type, schema in local.items():
                assert schema_signature(worker.schemas[node_type]) == schema_signature(schema)
            await worker.prepare(sorted(local))
            with pytest.raises(KeyError):
                await worker.prepare(["no.such.node"])
        finally:
            await worker.close()

    run_scenario(scenario())


def test_worker_converts_legacy_checkpoint_on_first_use(tmp_path: Path) -> None:
    import torch
    from dinkster_workers.legacy_sources import discover_converted_sidecar
    from safetensors.torch import load_file

    source = tmp_path / "weights.ckpt"
    torch.save({"weight": torch.tensor([3.0, -2.0])}, source)

    async def scenario() -> None:
        worker = IsolatedWorker(DEV_MANIFEST, core_registry(), aimdo_arm="off")
        try:
            await worker.start()
            assert worker.can_convert_legacy_checkpoint is True
            assert await worker.convert_legacy_checkpoint(source, "weights.ckpt") == (
                "success",
                None,
            )
            sidecar = discover_converted_sidecar(source)
            assert sidecar is not None
            assert load_file(sidecar)["weight"].tolist() == [3.0, -2.0]
        finally:
            await worker.close()

    run_scenario(scenario())


def test_isolated_run_matches_in_process_run() -> None:
    async def scenario() -> None:
        # In-process reference: pack types registered locally.
        local_registry = core_registry()
        register_dev_types(local_registry)
        local_engine = Engine(
            schemas=build_schemas(PACK_NODES),
            registry=local_registry,
            worker=InProcessWorker(build_node_types(PACK_NODES), local_registry),
            cache=MemoryLRUCache(),
        )
        reference = await local_engine.run(image_graph(), ["s", "b"])

        # Isolated: the parent registers only core types and never imports
        # the pack; schemas come from the handshake.
        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(image_graph(), ["s", "b"])
            # Same values, same fingerprints: cache keys are location-
            # independent (hazard H4).
            assert result.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
            for output_id in ("mean", "minimum", "maximum"):
                assert (
                    result.outputs["s"][output_id].fingerprint
                    == reference.outputs["s"][output_id].fingerprint
                )
            assert (
                result.outputs["b"]["image"].fingerprint
                == reference.outputs["b"]["image"].fingerprint
            )

            # Second run: cache hits, no boundary crossings needed.
            again = await engine.run(image_graph(), ["s", "b"])
            assert again.executed == ()
        finally:
            await worker.close()

    run_scenario(scenario())


def test_opaque_value_relays_back_across_the_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()  # iso.blob deliberately not registered here
        worker = iso_worker(tmp_path, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                nodes={
                    "out": GraphNode("iso.blob_out", {"size": 41}),
                    "len": GraphNode("iso.blob_len", {"blob": Link("out", "blob")}),
                }
            )
            result = await engine.run(graph, ["len"])
            # The blob crossed to the parent (opaque), was cached, and was
            # relayed back to the worker as its original codec bytes.
            assert result.outputs["len"]["length"].resolve() == 41
        finally:
            await worker.close()

    run_scenario(scenario())


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX signals")
def test_isolated_close_cancellation_still_kills_reaps_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_workers.isolated as isolated_module

    monkeypatch.setattr(isolated_module, "_CHILD_SHUTDOWN_GRACE", 0.05)

    async def scenario() -> None:
        worker = iso_worker(tmp_path, core_registry())
        await worker.start()
        process = worker._proc  # noqa: SLF001
        cleanup_dir = worker._tmpdir  # noqa: SLF001
        assert process is not None
        assert cleanup_dir is not None
        os.kill(process.pid, _SIGSTOP)

        waiting = asyncio.Event()
        original_wait = process.wait

        async def wait_for_exit() -> int:
            if asyncio.current_task() is not worker._process_watch:
                waiting.set()
            return await original_wait()

        monkeypatch.setattr(process, "wait", wait_for_exit)
        close = asyncio.create_task(worker.close())
        await asyncio.wait_for(waiting.wait(), 5)
        assert worker._close_task is not None and not worker._close_task.done()
        assert process.returncode is None
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close

        assert process.returncode == -_SIGKILL
        assert not cleanup_dir.exists()
        await worker.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    ("prompt", "extra_pnginfo"),
    [
        (
            {"save": {"class_type": "Save", "inputs": {"seed": 4}}},
            {"workflow": {"nodes": [4, 2]}},
        ),
        ({}, None),
        (None, {"workflow": {"nodes": [4, 2]}}),
    ],
)
def test_isolated_host_supplies_export_snapshot(
    tmp_path: Path,
    prompt: Mapping[str, object] | None,
    extra_pnginfo: Mapping[str, object] | None,
) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = iso_worker(tmp_path, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(nodes={"submitted-node-29": GraphNode("iso.execution_context", {})})
            snapshot = ExportSnapshot(
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
            result = await engine.run(graph, ["submitted-node-29"], export_snapshot=snapshot)
            assert result.outputs["submitted-node-29"]["snapshot"].resolve() == (
                f"submitted-node-29|{snapshot.prompt}|{snapshot.extra_pnginfo}"
            )
        finally:
            await worker.close()

    run_scenario(scenario())


def test_shm_transport_for_large_payloads_and_cleanup(shm_segments: set[str]) -> None:
    if not sys.platform.startswith(("linux",)):
        pytest.skip("shm leftovers check relies on /dev/shm")

    async def scenario() -> None:
        diagnostics: list[BoundaryDiagnostic] = []
        registry = core_registry()
        worker = IsolatedWorker(
            DEV_MANIFEST,
            registry,
            # Below the image payload size (~640 bytes: 16x8 float32 npy +
            # header), so images genuinely ride shm and the /dev/shm cleanup
            # assertion below actually exercises the handoff.
            shm_threshold=256,
            on_diagnostic=diagnostics.append,
        )
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(image_graph(), ["s"])
            assert result.outputs["s"]["mean"].resolve() == pytest.approx(0.5)

            by_node = {d.node_type: d for d in diagnostics}
            stats_inputs = {c.edge_id: c for c in by_node["dev.image.stats"].inputs}
            assert stats_inputs["image"].type_id == "dev.image"
            # Image payloads (~640 bytes) exceed the 256-byte threshold, so
            # they must cross via shared memory, not inline bytes.
            assert stats_inputs["image"].transport == "shm"
            blend_inputs = {c.edge_id: c for c in by_node["dev.image.blend"].inputs}
            # ratio is a small FLOAT: inline either way.
            assert blend_inputs["ratio"].transport == "inline"
        finally:
            await worker.close()
        assert shm_segments
        assert not [name for name in shm_segments if (Path("/dev/shm") / name).exists()]

    run_scenario(scenario())


def test_node_error_carries_message_and_worker_survives(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = iso_worker(tmp_path, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            boom = Graph(nodes={"n": GraphNode("iso.boom", {"tag": "t1"})})
            with pytest.raises(ExecutionError) as excinfo:
                await engine.run(boom, ["n"])
            assert "boom: t1" in excinfo.value.error.message
            assert "RuntimeError" in excinfo.value.error.traceback

            # The failure was the node's, not the boundary's: same worker
            # keeps serving.
            ok = Graph(nodes={"s": GraphNode("iso.sleepy", {"value": "hi", "seconds": 0.0})})
            result = await engine.run(ok, ["s"])
            assert result.outputs["s"]["value"].resolve() == "hi"
        finally:
            await worker.close()

    run_scenario(scenario())


def test_isolated_admission_rejects_before_execution_and_recovers() -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            for value in (5, "not a float"):
                message = (
                    "input 'value' literal does not match FLOAT"
                    if isinstance(value, str)
                    else "value must be a float literal"
                )
                with pytest.raises(GraphValidationError, match=message):
                    await asyncio.wait_for(
                        engine.run(
                            Graph(
                                {"check": GraphNode("dev.execution.admission", {"value": value})}
                            ),
                            ["check"],
                        ),
                        timeout=5,
                    )
            with pytest.raises(GraphValidationError, match="value must be linked from FLOAT"):
                await engine.run(
                    Graph(
                        {
                            "producer": GraphNode("dev.execution.fail"),
                            "check": GraphNode(
                                "dev.execution.admission", {"value": Link("producer", "value")}
                            ),
                        }
                    ),
                    ["check"],
                )
            result = await engine.run(
                Graph({"check": GraphNode("dev.execution.admission", {"value": 7.5})}),
                ["check"],
            )
            assert result.outputs["check"]["value"].resolve() == 7.5
        finally:
            await worker.close()

    run_scenario(scenario())


def test_isolated_lazy_hook_protocol_and_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = tmp_path / "lazy-cancel-entered"
    monkeypatch.setenv("ISOPACK_LAZY_CANCEL_ENTERED", str(entered))

    async def scenario() -> None:
        registry = core_registry()
        worker = iso_worker(tmp_path, registry)
        await worker.start()
        events: list[EngineEvent] = []
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=events.append,
            )

            def lazy_graph(
                mode: str,
                producer_type: str = "iso.sleepy",
                *,
                seconds: float = 0.0,
            ) -> Graph:
                producer_inputs: dict[str, object]
                if producer_type == "iso.boom":
                    producer_inputs = {"tag": "producer"}
                else:
                    producer_inputs = {"value": "selected", "seconds": seconds}
                return Graph(
                    {
                        "producer": GraphNode(producer_type, producer_inputs),
                        "lazy": GraphNode(
                            "iso.lazy_probe",
                            {"mode": mode, "value": Link("producer", "value")},
                        ),
                    }
                )

            result = await engine.run(lazy_graph("request"), ["lazy"])
            assert result.outputs["lazy"]["value"].resolve() == "selected"
            awaitable = await engine.run(lazy_graph("awaitable"), ["lazy"])
            assert awaitable.outputs["lazy"]["value"].resolve() == "selected"

            for mode, code in (
                ("malformed", "lazy-request-invalid"),
                ("awaitable-malformed", "lazy-request-invalid"),
                ("nonserializable", "lazy-request-invalid"),
                ("error", "lazy-hook-failed"),
                ("awaitable-error", "lazy-hook-failed"),
            ):
                with pytest.raises(ExecutionError) as exc_info:
                    await asyncio.wait_for(engine.run(lazy_graph(mode), ["lazy"]), timeout=2)
                assert exc_info.value.error.node_id == "lazy"
                assert code in exc_info.value.error.message
                if mode == "nonserializable":
                    assert exc_info.value.error.message == (
                        "lazy-request-invalid: requested input at index 0 must be a string"
                    )
                if mode == "awaitable-error":
                    assert "RuntimeError: async lazy hook exploded" in (
                        exc_info.value.error.traceback
                    )
                assert worker._session._lazy_pending == {}  # noqa: SLF001

            # A malformed hook result fails only its consumer. The worker
            # remains usable after returning the classified terminal error.
            after_refusal = await engine.run(lazy_graph("request"), ["lazy"])
            assert after_refusal.outputs["lazy"]["value"].resolve() == "selected"

            with pytest.raises(ExecutionError) as producer_error:
                await engine.run(lazy_graph("request", "iso.boom"), ["lazy"])
            assert producer_error.value.error.node_id == "producer"
            assert "boom: producer" in producer_error.value.error.message

            schema = worker.schemas["iso.lazy_probe"]

            def lazy_invocation(request_id: str, mode: str) -> LazyStatusInvocation:
                return LazyStatusInvocation(
                    request_id=request_id,
                    node_id="lazy",
                    node_type="iso.lazy_probe",
                    available_inputs={"mode": registry.wrap("STRING", mode)},
                    connected_undemanded_inputs=("value",),
                    effective_schema=schema,
                )

            cancelling = asyncio.create_task(
                worker.check_lazy_status(lazy_invocation("lazy-cancel", "awaitable-cancel"))
            )
            async with asyncio.timeout(5):
                while not entered.exists():
                    await asyncio.sleep(0.005)
            cancelling.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelling
            retry = await worker.check_lazy_status(lazy_invocation("lazy-retry", "cleanup-retry"))
            assert retry.error is None
            assert retry.requested_inputs == ("value",)
            assert worker._session._lazy_pending == {}  # noqa: SLF001

            slow = lazy_graph("request", seconds=30.0)
            events.clear()
            task = asyncio.create_task(engine.run(slow, ["lazy"]))
            async with asyncio.timeout(5):
                while not any(
                    event.kind == "node_event" and event.detail.get("name") == "lazy_demand"
                    for event in events
                ):
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not any(event.kind in {"node_failed", "run_finished"} for event in events)
        finally:
            await worker.close()

    run_scenario(scenario())


@pytest.mark.parametrize("selected", (False, True))
def test_isolated_switch_executes_only_the_selected_input(selected: bool) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(
                {
                    "first": GraphNode("dev.execution.value", {"value": 17}),
                    "second": GraphNode("dev.execution.value", {"value": 43}),
                    "route": GraphNode(
                        "dev.execution.lazy_select",
                        {
                            "choose_right": selected,
                            "left": Link("first", "value"),
                            "right": Link("second", "value"),
                        },
                    ),
                }
            )

            result = await engine.run(graph, ["route"])
            expected = "second" if selected else "first"
            assert result.outputs["route"]["value"].resolve() == (43 if selected else 17)
            assert set(result.executed) == {expected, "route"}
        finally:
            await worker.close()

    run_scenario(scenario())


def test_worker_crash_fails_cleanly(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = iso_worker(tmp_path, registry)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            assert isinstance(worker.instance_token, str) and worker.instance_token
            crash = Graph(nodes={"n": GraphNode("iso.exit", {"code": 3})})
            with pytest.raises(ExecutionError) as excinfo:
                await engine.run(crash, ["n"])
            assert "isopack" in excinfo.value.error.message
            assert "not running" in excinfo.value.error.message
            assert "exit code 3" in excinfo.value.error.message
            # The lifetime token died with the session (read-loop death, not
            # an orderly close): a resolver can never match a dead lifetime.
            assert not worker.alive
            assert worker.instance_token is None

            # Dead worker fails fast, never hangs.
            after = Graph(nodes={"s": GraphNode("iso.sleepy", {"value": "x"})})
            with pytest.raises(ExecutionError, match="not running") as after_exc:
                await engine.run(after, ["s"])
            assert "exit code 3" in after_exc.value.error.message
        finally:
            await worker.close()

    run_scenario(scenario())


def test_cancellation_reaches_the_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = iso_worker(tmp_path, registry)
        await worker.start()
        try:
            entered = asyncio.Event()

            def on_event(event: EngineEvent) -> None:
                if event.kind == "node_event" and event.detail.get("name") == "progress":
                    entered.set()

            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
                on_event=on_event,
            )
            slow = Graph(nodes={"s": GraphNode("iso.chatty", {"value": "never", "seconds": 30.0})})
            task = asyncio.create_task(engine.run(slow, ["s"]))
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # The child cancelled the node task; the worker stays healthy.
            quick = Graph(nodes={"s": GraphNode("iso.sleepy", {"value": "ok", "seconds": 0.0})})
            result = await engine.run(quick, ["s"])
            assert result.outputs["s"]["value"].resolve() == "ok"
        finally:
            await worker.close()

    run_scenario(scenario())


def test_routing_worker_mixes_isolated_and_in_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        register_dev_types(registry)

        isolated = iso_worker(tmp_path, registry)
        await isolated.start()
        try:
            local = InProcessWorker(build_node_types(PACK_NODES), registry)
            routes = {node_type: isolated for node_type in isolated.schemas}
            worker = RoutingWorker(routes, default=local)

            schemas = {**build_schemas(PACK_NODES), **isolated.schemas}
            engine = Engine(
                schemas=schemas, registry=registry, worker=worker, cache=MemoryLRUCache()
            )
            # A single graph spanning both placements - zero engine changes.
            graph = Graph(
                nodes={
                    "hello": GraphNode("dev.util.delay", {"value": "hello", "seconds": 0.0}),
                    "echo": GraphNode("iso.sleepy", {"value": Link("hello", "value")}),
                }
            )
            result = await engine.run(graph, ["echo"])
            assert result.outputs["echo"]["value"].resolve() == "hello"
        finally:
            await isolated.close()

    run_scenario(scenario())


@pytest.mark.skipif(
    not os.environ.get("DINKSTER_SLOW_TESTS") or shutil.which("uv") is None,
    reason="venv provisioning is slow; set DINKSTER_SLOW_TESTS=1 (requires uv)",
)
def test_pack_runs_in_its_own_provisioned_venv(tmp_path: Path) -> None:
    """The full M2 claim: the pack executes in a venv the parent's
    interpreter has never seen, provisioned from the manifest."""
    manifest = load_manifest(DEV_MANIFEST)
    workspace = [
        REPO_ROOT / "packages" / name
        for name in (
            "dinkster-workers",
            "dinkster-protocol",
            "dinkster-schema",
            "dinkster-values",
            "dinkster-api",
            "dinkster-memory",
            "dinkster-assets",
            "dinkster-caches",
            "dinkster-inference-wire",
            "dinkster-video-runtime",
            "dinkster-image-document",
        )
    ]
    python = ensure_pack_venv(manifest, venv_root=tmp_path / "venvs", workspace_packages=workspace)

    # The layering payoff, asserted in the child interpreter itself: the
    # boundary package is importable, the scheduler is NOT installed. An
    # accidental future dependency edge that drags dinkster-engine (or graph)
    # back into pack venvs must fail here, not go unnoticed.
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util as u\n"
            "assert u.find_spec('dinkster_protocol') is not None, 'protocol missing'\n"
            "assert u.find_spec('dinkster_engine') is None, 'scheduler leaked into child venv'\n"
            "assert u.find_spec('dinkster_graph') is None, 'graph leaked into child venv'\n",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr

    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(DEV_MANIFEST, registry, python=str(python))
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            result = await engine.run(image_graph(), ["s"])
            assert result.outputs["s"]["mean"].resolve() == pytest.approx(0.5)
        finally:
            await worker.close()

    run_scenario(scenario())


def test_manifest_validation(tmp_path: Path) -> None:
    good = tmp_path / "ok.toml"
    good.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    manifest = load_manifest(good)
    assert manifest.name == "p"
    assert manifest.nodes_entry == "m:N"
    assert manifest.types_entry is None
    assert manifest.requires == ()
    assert manifest.root == tmp_path

    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "missing.toml")

    bad_entry = tmp_path / "bad_entry.toml"
    bad_entry.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "no-colon"\n')
    with pytest.raises(ManifestError, match="module:attr"):
        load_manifest(bad_entry)

    no_name = tmp_path / "no_name.toml"
    no_name.write_text('[pack]\n[pack.entry]\nnodes = "m:N"\n')
    with pytest.raises(ManifestError, match="name"):
        load_manifest(no_name)

    bad_name = tmp_path / "bad_name.toml"
    bad_name.write_text('[pack]\nname = "My-Pack"\n[pack.entry]\nnodes = "m:N"\n')
    with pytest.raises(ManifestError, match="lowercase"):
        load_manifest(bad_name)


def test_manifest_namespaces(tmp_path: Path) -> None:
    """[pack] namespaces: the pack's node-type claims (DESIGN M8).
    Grammar-valid and mutually non-overlapping, fatal when malformed (a
    dropped claim would just fail composition later, with a worse
    message); absent means the pack claims its own name."""

    def manifest(body: str) -> Path:
        path = tmp_path / "dinkster-pack.toml"
        path.write_text(f'[pack]\nname = "p"\n{body}[pack.entry]\nnodes = "m:N"\n')
        return path

    assert load_manifest(manifest("")).namespaces == ("p",)
    assert load_manifest(manifest('namespaces = ["img", "audio.fx"]\n')).namespaces == (
        "img",
        "audio.fx",
    )
    # Reserved roots are legal manifest SHAPE - trust is composition
    # policy, not loader policy (core packs declare them like anyone).
    assert load_manifest(manifest('namespaces = ["std"]\n')).namespaces == ("std",)

    # An explicit empty list claims nothing - legal only for a pure
    # executor (non-empty [pack] executes), so the shape cannot leave a
    # pack's own node types silently uncovered.
    assert (
        load_manifest(manifest('namespaces = []\nexecutes = ["training.advance"]\n')).namespaces
        == ()
    )

    with pytest.raises(ManifestError, match="list of strings"):
        load_manifest(manifest('namespaces = "img"\n'))
    with pytest.raises(ManifestError, match=r"requires \[pack\] executes"):
        load_manifest(manifest("namespaces = []\n"))
    with pytest.raises(ManifestError, match="list of strings"):
        load_manifest(manifest("namespaces = [3]\n"))
    assert load_manifest(manifest('namespaces = ["Img"]\n')).namespaces == ("Img",)
    uppercase_pack = manifest('namespaces = ["Img"]\n')
    uppercase_pack.write_text(uppercase_pack.read_text().replace('name = "p"', 'name = "P"'))
    with pytest.raises(ManifestError, match="lowercase"):
        load_manifest(uppercase_pack)
    # Separator equivalence: img_x and img.x are the SAME claim.
    with pytest.raises(ManifestError, match="same claim"):
        load_manifest(manifest('namespaces = ["img_x", "img.x"]\n'))
    with pytest.raises(ManifestError, match="overlap"):
        load_manifest(manifest('namespaces = ["img", "img.filters"]\n'))


def test_manifest_extra_requires(tmp_path: Path) -> None:
    """[pack.extra-requires]: accelerator-conditional dependencies - the
    one conditional dimension PEP 508 markers cannot express. Unknown
    accelerator keys are fatal (they would be silently dead on every
    host); requirement strings pass through verbatim, markers included."""

    def manifest(body: str) -> Path:
        path = tmp_path / "dinkster-pack.toml"
        path.write_text(
            f'[pack]\nname = "p"\nrequires = ["numpy"]\n[pack.entry]\nnodes = "m:N"\n{body}'
        )
        return path

    plain = load_manifest(manifest(""))
    assert plain.extra_requires == ()  # omitted = base requires everywhere
    assert plain.requires_for("cuda") == ("numpy",)

    loaded = load_manifest(
        manifest(
            "[pack.extra-requires]\n"
            'rocm = ["torch==2.5.1+rocm6.2"]\n'
            'cuda = ["torch==2.5.1", "nvidia-ml-py; sys_platform == \'linux\'"]\n'
            "cpu = []\n"
        )
    )
    # entries sorted by key; requirement lists verbatim (markers intact, order kept)
    assert loaded.extra_requires == (
        ("cpu", ()),
        ("cuda", ("torch==2.5.1", "nvidia-ml-py; sys_platform == 'linux'")),
        ("rocm", ("torch==2.5.1+rocm6.2",)),
    )
    assert loaded.requires_for("cuda") == (
        "numpy",
        "torch==2.5.1",
        "nvidia-ml-py; sys_platform == 'linux'",
    )
    assert loaded.requires_for("rocm") == ("numpy", "torch==2.5.1+rocm6.2")
    assert loaded.requires_for("cpu") == ("numpy",)
    assert loaded.requires_for("mps") == ("numpy",)  # undeclared accelerator = base

    with pytest.raises(ManifestError, match="not a known accelerator"):
        load_manifest(manifest('[pack.extra-requires]\ngpu = ["torch"]\n'))
    not_a_table = tmp_path / "not-a-table.toml"
    not_a_table.write_text(
        '[pack]\nname = "p"\nextra-requires = ["torch"]\n[pack.entry]\nnodes = "m:N"\n'
    )
    with pytest.raises(ManifestError, match="must be a table"):
        load_manifest(not_a_table)
    with pytest.raises(ManifestError, match="list of requirement"):
        load_manifest(manifest('[pack.extra-requires]\ncuda = "torch"\n'))
    with pytest.raises(ManifestError, match="list of requirement"):
        load_manifest(manifest("[pack.extra-requires]\ncuda = [3]\n"))


def test_ensure_group_venv_retries_after_partial_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import provision

    root = tmp_path / "alpha"
    root.mkdir()
    path = root / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "alpha"\nrequires = ["numpy"]\n[pack.entry]\nnodes = "alpha:N"\n'
    )
    manifest = load_manifest(path)
    installs = 0

    def fail_first(command: list[str]) -> None:
        nonlocal installs
        if command[1] == "venv":
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!fake\n")
            return
        installs += 1
        if installs == 1:
            raise provision.ProvisionError("conflict")

    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", fail_first)
    venv_root = tmp_path / "venvs"
    with pytest.raises(provision.ProvisionError, match="conflict"):
        provision.ensure_group_venv([manifest], "models", venv_root=venv_root)
    assert not (venv_root / "models").exists()
    provision.ensure_group_venv([manifest], "models", venv_root=venv_root)
    assert installs == 2
    assert (venv_root / "models" / ".dinkster-complete").is_file()


def test_ensure_pack_venv_retries_after_partial_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import provision

    root = tmp_path / "alpha"
    root.mkdir()
    path = root / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "alpha"\nrequires = ["numpy"]\n[pack.entry]\nnodes = "alpha:N"\n'
    )
    manifest = load_manifest(path)
    installs = 0

    def fail_first(command: list[str]) -> None:
        nonlocal installs
        if command[1] == "venv":
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!fake\n")
            return
        installs += 1
        if installs == 1:
            raise provision.ProvisionError("conflict")

    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", fail_first)
    venv_root = tmp_path / "venvs"
    with pytest.raises(provision.ProvisionError, match="conflict"):
        provision.ensure_pack_venv(manifest, venv_root=venv_root)
    assert not (venv_root / "alpha").exists()
    provision.ensure_pack_venv(manifest, venv_root=venv_root)
    assert installs == 2
    assert (venv_root / "alpha" / ".dinkster-complete").is_file()


def test_pack_icon_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """validate_pack_icon enforces the frontend contract: exactly 64x64,
    static PNG/WebP sniffed from bytes, <= 64 KiB, contained in the pack
    directory. Valid icons carry sniffed media type, sha256 digest, and
    the digested bytes themselves."""
    from dinkster_workers import validate_pack_icon
    from dinkster_workers.manifest import ICON_MAX_BYTES
    from icon_bytes import animated_webp_bytes, png_bytes, webp_bytes

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n')

    (tmp_path / "icon.png").write_bytes(png_bytes())
    icon, problem = validate_pack_icon(manifest_path, "icon.png")
    assert problem is None and icon is not None
    assert icon.media_type == "image/png"
    assert icon.digest.startswith("sha256:") and len(icon.digest) == 71
    assert icon.data == png_bytes()

    # Media type is sniffed from the bytes, never the extension.
    (tmp_path / "actually-webp.png").write_bytes(webp_bytes())
    icon, problem = validate_pack_icon(manifest_path, "actually-webp.png")
    assert problem is None and icon is not None
    assert icon.media_type == "image/webp"

    def rejected(name: str, data: bytes) -> str:
        (tmp_path / name).write_bytes(data)
        icon, problem = validate_pack_icon(manifest_path, name)
        assert icon is None and problem is not None
        return problem

    assert "64x64" in rejected("wrong-size.png", png_bytes(32, 32))
    assert "64x64" in rejected("wrong-size.webp", webp_bytes(65, 64))
    assert "static" in rejected("apng.png", png_bytes(animated=True))
    assert "static" in rejected("anim.webp", animated_webp_bytes())
    assert "PNG or WebP" in rejected("not-image.png", b"GIF89a" + b"\x00" * 64)
    assert "PNG or WebP" in rejected("truncated.png", png_bytes()[:20])
    oversized_path = tmp_path / "huge.png"
    oversized_path.write_bytes(png_bytes() + b"\x00" * ICON_MAX_BYTES)
    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def record_read(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    icon, problem = validate_pack_icon(manifest_path, oversized_path.name)
    assert icon is None and problem is not None and "bytes" in problem
    assert oversized_path not in read_paths

    growing_path = tmp_path / "growing.png"
    growing_path.write_bytes(png_bytes())

    def grow_after_stat(path: Path) -> bytes:
        if path == growing_path:
            return png_bytes() + b"\x00" * ICON_MAX_BYTES
        return record_read(path)

    monkeypatch.setattr(Path, "read_bytes", grow_after_stat)
    icon, problem = validate_pack_icon(manifest_path, growing_path.name)
    assert icon is None and problem is not None and "bytes" in problem

    # Containment: absolute paths, traversal, and symlink escape all fail.
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(png_bytes())
    _, problem = validate_pack_icon(manifest_path, str(outside))
    assert problem is not None and "relative" in problem
    _, problem = validate_pack_icon(manifest_path, "../outside.png")
    assert problem is not None and "relative" in problem
    symlink_or_skip(tmp_path / "sneaky.png", outside)
    _, problem = validate_pack_icon(manifest_path, "sneaky.png")
    assert problem is not None and "escape" in problem

    _, problem = validate_pack_icon(manifest_path, "missing.png")
    assert problem is not None and "unreadable" in problem


def test_manifest_rejects_docs_root_traversal_and_symlinks(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_prefix = '[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n[pack.docs]\n'
    outside = tmp_path.parent / f"{tmp_path.name}-outside-docs"
    outside.mkdir()

    manifest_path.write_text(manifest_prefix + f'dir = "../{outside.name}"\n')
    traversal = load_manifest(manifest_path)
    assert traversal.docs is None
    assert traversal.docs_problem is not None and "relative" in traversal.docs_problem

    docs_target = tmp_path / "docs-target"
    docs_target.mkdir()
    docs_link = tmp_path / "docs-link"
    try:
        docs_link.symlink_to(docs_target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    manifest_path.write_text(manifest_prefix + 'dir = "docs-link"\n')
    root_link = load_manifest(manifest_path)
    assert root_link.docs is None
    assert root_link.docs_problem is not None and "symlink" in root_link.docs_problem

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "linked.md").symlink_to(outside / "page.md")
    manifest_path.write_text(manifest_prefix + 'dir = "docs"\n')
    nested_link = load_manifest(manifest_path)
    assert nested_link.docs is None
    assert nested_link.docs_problem is not None and "symlink" in nested_link.docs_problem


def test_manifest_drops_bad_locale_files_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers.manifest import LOCALE_CATALOG_MAX_BYTES

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    locales = tmp_path / "locales"
    locales.mkdir()
    (locales / "en.json").write_text('{"nodes":{"p.echo":{"displayName":"Echo"}}}\n')
    (locales / "PT-br.json").write_text("{}")
    (locales / "bad.json").write_bytes(b"\xff")
    (locales / "fr.json").write_text("{")
    (locales / "zh.json").write_bytes(b" " * (LOCALE_CATALOG_MAX_BYTES + 1))
    (locales / "nested").mkdir()
    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    manifest = load_manifest(manifest_path)

    assert [catalog.locale for catalog in manifest.locale_catalogs] == ["en"]
    assert len(manifest.locale_catalog_problems) == 5
    assert any("canonical lowercase" in problem for problem in manifest.locale_catalog_problems)
    assert sum("invalid JSON" in problem for problem in manifest.locale_catalog_problems) == 2
    assert any("byte cap" in problem for problem in manifest.locale_catalog_problems)
    assert any("directly beneath" in problem for problem in manifest.locale_catalog_problems)
    assert locales / "zh.json" not in read_paths


def test_manifest_rejects_locale_catalog_file_symlinks(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    locales = tmp_path / "locales"
    locales.mkdir()
    try:
        (locales / "en.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    manifest = load_manifest(manifest_path)

    assert manifest.locale_catalogs == ()
    assert len(manifest.locale_catalog_problems) == 1
    assert "symlink" in manifest.locale_catalog_problems[0]


def test_validate_pack_blueprint(tmp_path: Path) -> None:
    """validate_pack_blueprint enforces the backend-ownable half of the
    frontend contract: id grammar, name/tags shape, path containment,
    <= 1 MiB, UTF-8, well-formed JSON with an object at the top level,
    sha256 digest over the exact file bytes. Never document semantics -
    the workflow format is frontend-owned."""
    import hashlib

    from dinkster_workers import validate_pack_blueprint
    from dinkster_workers.manifest import BLUEPRINT_MAX_BYTES

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n')

    data = b'{"graphs": {"main": {}}, "meta": {"name": "Upscale"}}'
    (tmp_path / "upscale.json").write_bytes(data)
    entry = {
        "id": "upscale",
        "name": "Upscale",
        "description": "Starter upscale workflow",
        "tags": ["image", "upscale"],
        "file": "upscale.json",
    }
    blueprint, problem = validate_pack_blueprint(manifest_path, entry)
    assert problem is None and blueprint is not None
    assert blueprint.id == "upscale"
    assert blueprint.name == "Upscale"
    assert blueprint.description == "Starter upscale workflow"
    assert blueprint.tags == ("image", "upscale")
    assert blueprint.digest == "sha256:" + hashlib.sha256(data).hexdigest()
    assert blueprint.data == data

    # description and tags are optional; omission is empty, never None.
    minimal, problem = validate_pack_blueprint(
        manifest_path, {"id": "upscale", "name": "Upscale", "file": "upscale.json"}
    )
    assert problem is None and minimal is not None
    assert minimal.description == "" and minimal.tags == ()
    assert minimal.boundary_inputs == () and minimal.boundary_outputs == ()

    # boundary_inputs/boundary_outputs are author-declared search hints
    # passed through VERBATIM: shape-checked (list of non-empty strings)
    # but never resolved against the type registry or the document - the
    # backend cannot see whether "no.such.type" exists, and must not try.
    hinted, problem = validate_pack_blueprint(
        manifest_path,
        {
            "id": "upscale",
            "name": "Upscale",
            "file": "upscale.json",
            "boundary_inputs": ["core.image", "no.such.type"],
            "boundary_outputs": ["core.image"],
        },
    )
    assert problem is None and hinted is not None
    assert hinted.boundary_inputs == ("core.image", "no.such.type")
    assert hinted.boundary_outputs == ("core.image",)

    def rejected(entry: object) -> str:
        blueprint, problem = validate_pack_blueprint(manifest_path, entry)
        assert blueprint is None and problem is not None
        return problem

    assert "table" in rejected("not-a-table")
    assert "'id'" in rejected({"name": "X", "file": "upscale.json"})
    assert "lowercase" in rejected({"id": "Upscale!", "name": "X", "file": "upscale.json"})
    assert "'name'" in rejected({"id": "x", "file": "upscale.json"})
    assert "'tags'" in rejected({"id": "x", "name": "X", "file": "upscale.json", "tags": [1]})
    assert "'boundary_inputs'" in rejected(
        {"id": "x", "name": "X", "file": "upscale.json", "boundary_inputs": [1]}
    )
    assert "'boundary_outputs'" in rejected(
        {"id": "x", "name": "X", "file": "upscale.json", "boundary_outputs": [""]}
    )
    assert "'file'" in rejected({"id": "x", "name": "X"})
    assert "unreadable" in rejected({"id": "x", "name": "X", "file": "missing.json"})

    # Containment: absolute paths, traversal, and symlink escape all fail.
    outside = tmp_path.parent / "outside.json"
    outside.write_bytes(data)
    assert "relative" in rejected({"id": "x", "name": "X", "file": str(outside)})
    assert "relative" in rejected({"id": "x", "name": "X", "file": "../outside.json"})
    symlink_or_skip(tmp_path / "sneaky.json", outside)
    assert "escape" in rejected({"id": "x", "name": "X", "file": "sneaky.json"})

    # Byte cap, UTF-8, JSON well-formedness, top-level object.
    (tmp_path / "huge.json").write_bytes(b'{"k": "' + b"a" * BLUEPRINT_MAX_BYTES + b'"}')
    assert "cap" in rejected({"id": "x", "name": "X", "file": "huge.json"})
    (tmp_path / "at-cap.json").write_bytes(b'{"k": "' + b"a" * (BLUEPRINT_MAX_BYTES - 9) + b'"}')
    _, problem = validate_pack_blueprint(
        manifest_path, {"id": "x", "name": "X", "file": "at-cap.json"}
    )
    assert problem is None  # exactly at the cap is fine
    (tmp_path / "not-utf8.json").write_bytes(b'{"k": "\xff\xfe"}')
    assert "UTF-8" in rejected({"id": "x", "name": "X", "file": "not-utf8.json"})
    (tmp_path / "not-json.json").write_bytes(b"{nope")
    assert "JSON" in rejected({"id": "x", "name": "X", "file": "not-json.json"})
    (tmp_path / "array.json").write_bytes(b"[1, 2]")
    assert "object" in rejected({"id": "x", "name": "X", "file": "array.json"})
