"""M2: the boundary proves itself. An isolated worker process hosts a pack;
the engine cannot tell (hazard H3), fingerprints are location-independent
(H4), the parent never imports pack code (H1/H5), and the crossing's cost
is visible as diagnostics (DESIGN 3.9)."""

from __future__ import annotations

import asyncio
import glob
import importlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent, ExecutionError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import DEFAULT_INFERENCE_RESERVE_BYTES, InvocationView, MemoryGovernor
from dinkster_nodes_dev import PACK_NODES
from dinkster_protocol import (
    GRAPH_COMPILE_CANCEL_TYPE,
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_REQUEST_TYPE,
    GRAPH_COMPILE_RESULT_TYPE,
    WORKGROUP_CAPABILITY,
    AttentionCapabilityEvidence,
    AttentionPolicyConfig,
    AttentionRouteToken,
    DeviceResourceId,
    ExportSnapshot,
    Invocation,
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
    attention_capability_evidence_to_wire,
    attention_route_token_to_wire,
    derive_attention_route_token,
)
from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    ComfyAliasConfidence,
    ComfyAliasRecord,
    ComfyAliasRegistry,
    ComfyAliasSource,
    ComfyAliasSourceSchema,
    ComfyGroupRegistry,
    InputSpec,
    MappingSource,
    NodeSchema,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    build_node_types,
    build_schemas,
    comfy_alias_registry_to_wire,
    comfy_group_registry_to_wire,
    schema_signature,
)
from dinkster_values import TypeRegistry, UnresolvablePayload, register_core_types
from dinkster_workers import (
    AcceleratorError,
    BoundaryDiagnostic,
    DeviceMap,
    GroupIsolatedWorker,
    HeadroomMirror,
    InProcessWorker,
    IsolatedWorker,
    ManifestError,
    RoutingWorker,
    detect_accelerator,
    detect_runtime,
    ensure_pack_venv,
    load_manifest,
    resolve_accelerator,
)
from dinkster_workers import host as host_module
from dinkster_workers.boundary import (
    DEFAULT_SHM_THRESHOLD,
    ValueCodec,
    encode_invocation,
    read_frame,
    write_frame,
)
from dinkster_workers.doctor import diagnose
from dinkster_workers.isolated import GroupMemberWorker
from dinkster_workers.launch import HOST_OWNED_ENVIRONMENT, Launcher, LaunchSpec
from dinkster_workers.session import BoundarySession, WorkerDied
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

from tests.platform_support import symlink_or_skip

REPO_ROOT = Path(__file__).parents[1]
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SIGSTOP = getattr(signal, "SIGSTOP", signal.SIGTERM)
_KILLED_RETURN_CODE = 1 if os.name == "nt" else -_SIGKILL
# Boundary tests exercise the numpy image family, which lives in the DEV
# pack (test scaffolding, never a user surface) - so the isolated worker
# under test hosts dinkster-nodes-dev.
DEV_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"
FOUNDATION_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-foundation" / "dinkster-pack.toml"
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
        f'[pack]\nname = "{name}"\n\n[pack.entry]\n'
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


def test_ordinary_isolated_worker_keeps_workgroup_capability_absent(tmp_path: Path) -> None:
    async def scenario() -> None:
        worker = iso_worker(tmp_path, core_registry())
        await worker.start()
        try:
            assert worker.workgroup_capabilities == frozenset()
        finally:
            await worker.close()

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_isolated_worker_uses_interpreter_entry_for_split_package(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    (pack_root / "isopack_nodes").mkdir()
    (pack_root / "isopack_nodes" / "__init__.py").write_text("\n")
    manifest = pack_root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "isopack"\n\n[pack.entry]\n'
        'nodes = "isopack_nodes.entries:NODES"\n'
        'types = "isopack_nodes.entries:register_types"\n'
    )
    interpreter_root = tmp_path / "interpreter"
    installed_package = interpreter_root / "isopack_nodes"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("\n")
    shutil.copy(TESTS_DIR / "isopack_nodes.py", installed_package / "entries.py")

    async def scenario() -> None:
        worker = IsolatedWorker(
            manifest,
            core_registry(),
            extra_env={"PYTHONPATH": str(interpreter_root)},
        )
        await worker.start()
        try:
            assert "iso.sleepy" in worker.schemas
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_launched_worker_announces_comfy_translation_registries(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        aliases = ComfyAliasRegistry(source_schemas=(), records=())
        groups = ComfyGroupRegistry(source_schemas=(), group_schemas=(), records=())
        (tmp_path / "comfy-aliases.json").write_text(
            json.dumps(comfy_alias_registry_to_wire(aliases)),
            encoding="utf-8",
        )
        (tmp_path / "comfy-groups.json").write_text(
            json.dumps(comfy_group_registry_to_wire(groups)),
            encoding="utf-8",
        )
        worker = IsolatedWorker(
            manifest,
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        await worker.start()
        try:
            assert worker._session.comfy_aliases == aliases  # noqa: SLF001
            assert worker._session.comfy_groups == groups  # noqa: SLF001
        finally:
            await worker.close()

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_group_validates_every_workgroup_handler_before_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        manifests = (
            write_workgroup_manifest(first),
            write_workgroup_manifest(second, factory="invalid_workgroup_handler_factory"),
        )
        connected: list[str] = []

        async def unexpected_connect(endpoint: str):
            connected.append(endpoint)
            raise AssertionError("boundary opened before every handler validated")

        monkeypatch.setattr(host_module, "connect_endpoint", unexpected_connect)
        with pytest.raises(ManifestError, match="must return a callable handler"):
            await host_module.serve_many(
                ("first", "second"),
                tuple(str(path) for path in manifests),
                shm_threshold=DEFAULT_SHM_THRESHOLD,
                use_shm=False,
            )
        assert connected == []

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def graph_compile_session() -> BoundarySession:
    registry = core_registry()
    session = BoundarySession(
        registry,
        role="test worker",
        pack="test",
        codec=ValueCodec(registry),
    )
    session._schemas = {}  # noqa: SLF001 - construct a negotiated test session
    session._alive = True  # noqa: SLF001
    return session


async def start_test_host() -> tuple[
    asyncio.StreamReader,
    asyncio.StreamWriter,
    asyncio.Server,
    asyncio.Task[None],
]:
    registry = core_registry()
    worker = InProcessWorker({}, registry)
    accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        accepted.set_result(task)
        await host_module.serve_connection(
            reader,
            writer,
            pack_name="test",
            worker=worker,
            schemas={},
            planner=None,
            consumers={},
            codec=ValueCodec(registry),
        )

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()
    reader, writer = await asyncio.open_connection(address[0], address[1])
    host_task = await accepted
    hello = await read_frame(reader)
    assert hello is not None and hello[0]["type"] == "hello"
    return reader, writer, server, host_task


async def stop_test_host(
    writer: asyncio.StreamWriter,
    server: asyncio.Server,
    host_task: asyncio.Task[None],
) -> None:
    await write_frame(writer, {"type": "shutdown"}, [])
    await host_task
    server.close()
    await server.wait_closed()


def test_boundary_session_strictly_authenticates_attention_hello_evidence() -> None:
    async def negotiate(
        raw_token: object | None, raw_capabilities: object | None = None
    ) -> tuple[AttentionRouteToken | None, AttentionCapabilityEvidence | None]:
        accepted: asyncio.Future[asyncio.StreamWriter] = asyncio.get_running_loop().create_future()

        async def send_hello(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            accepted.set_result(writer)
            hello: dict[str, object] = {
                "type": "hello",
                "pack": "attention-test",
                "schemas": {},
            }
            if raw_token is not None:
                hello["attentionRouteToken"] = raw_token
            if raw_capabilities is not None:
                hello["attentionCapabilities"] = raw_capabilities
            await write_frame(writer, hello, [])

        server = await asyncio.start_server(send_hello, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()
        reader, writer = await asyncio.open_connection(address[0], address[1])
        peer_writer = await accepted
        session = BoundarySession(
            core_registry(),
            role="test worker",
            pack="test",
            codec=ValueCodec(core_registry()),
        )
        try:
            await session.begin(reader, writer, timeout=2)
            return session.attention_route_token, session.attention_capabilities
        finally:
            await session.close()
            peer_writer.close()
            await peer_writer.wait_closed()
            server.close()
            await server.wait_closed()

    token = attention_token()
    capabilities = attention_capabilities()
    assert asyncio.run(negotiate(attention_route_token_to_wire(token))) == (token, None)
    assert asyncio.run(
        negotiate(
            attention_route_token_to_wire(token),
            attention_capability_evidence_to_wire(capabilities),
        )
    ) == (token, capabilities)
    malformed = attention_route_token_to_wire(token)
    malformed["forged"] = True
    with pytest.raises(RuntimeError, match="malformed attentionRouteToken"):
        asyncio.run(negotiate(malformed))
    malformed_capabilities = attention_capability_evidence_to_wire(capabilities)
    malformed_capabilities["forged"] = True
    with pytest.raises(RuntimeError, match="malformed attentionCapabilities"):
        asyncio.run(negotiate(attention_route_token_to_wire(token), malformed_capabilities))
    with pytest.raises(RuntimeError, match="without attentionRouteToken"):
        asyncio.run(negotiate(None, attention_capability_evidence_to_wire(capabilities)))
    with pytest.raises(RuntimeError, match="inconsistent with attentionRouteToken"):
        asyncio.run(
            negotiate(
                attention_route_token_to_wire(attention_token(torch_version="forged")),
                attention_capability_evidence_to_wire(capabilities),
            )
        )


def test_host_requires_exact_optional_attention_evidence_before_reservation_planning() -> None:
    async def scenario() -> None:
        registry = core_registry()
        startup_token = attention_token()
        forged = attention_token(torch_version="forged")

        async def attempt(
            worker_token: AttentionRouteToken | None,
            invocation_token: AttentionRouteToken | None,
            planner_expected: bool,
        ) -> None:
            worker = InProcessWorker(
                {},
                registry,
                attention_capabilities=(
                    attention_capabilities() if worker_token is not None else None
                ),
                attention_route_token=worker_token,
            )
            planner_calls: list[InvocationView] = []
            accepted: asyncio.Future[asyncio.Task[None]] = (
                asyncio.get_running_loop().create_future()
            )

            async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                task = asyncio.current_task()
                assert task is not None
                accepted.set_result(task)
                await host_module.serve_connection(
                    reader,
                    writer,
                    pack_name="test",
                    worker=worker,
                    schemas={},
                    planner=lambda invocation: planner_calls.append(invocation) or (),
                    consumers={},
                    codec=ValueCodec(registry),
                )

            server = await asyncio.start_server(accept, "127.0.0.1", 0)
            address = server.sockets[0].getsockname()
            reader, writer = await asyncio.open_connection(address[0], address[1])
            host_task = await accepted
            hello = await read_frame(reader)
            assert hello is not None and hello[0]["type"] == "hello"
            if worker_token is None:
                assert "attentionRouteToken" not in hello[0]
                assert "attentionCapabilities" not in hello[0]
            else:
                assert hello[0]["attentionRouteToken"] == attention_route_token_to_wire(
                    worker_token
                )
                assert hello[0]["attentionCapabilities"] == (
                    attention_capability_evidence_to_wire(attention_capabilities())
                )
            invocation = Invocation(
                invocation_id="evidence-matrix",
                node_id="evidence-matrix",
                node_type="test.missing",
                inputs={},
                effective_schema=NodeSchema(node_type="test.missing"),
                attention_route_token=invocation_token,
            )
            header, blobs, segments, _stats = encode_invocation(ValueCodec(registry), invocation)
            assert segments == []
            await write_frame(writer, header, blobs)
            result = await asyncio.wait_for(read_frame(reader), timeout=2)
            assert result is not None
            assert bool(planner_calls) is planner_expected
            if not planner_expected:
                assert result[0]["error"]["message"] == (
                    "attention route token does not match worker startup evidence"
                )
            await stop_test_host(writer, server, host_task)

        await attempt(None, None, True)
        await attempt(startup_token, startup_token, True)
        await attempt(startup_token, None, False)
        await attempt(None, startup_token, False)
        await attempt(startup_token, forged, False)

    asyncio.run(scenario())


def test_aimdo_bootstrap_disabled_does_not_import_or_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def unexpected_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    caplog.clear()
    assert host_module._bootstrap_aimdo(False) is False
    assert caplog.records == []


def test_aimdo_bootstrap_unavailable_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def unavailable(name: str) -> object:
        assert name == "dinkster_aimdo.control"
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    assert host_module._bootstrap_aimdo(True) is False
    assert "continues without successful aimdo bootstrap" in caplog.text


def test_aimdo_bootstrap_false_result_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    control = SimpleNamespace(init=lambda: False)
    monkeypatch.setattr(importlib, "import_module", lambda name: control)
    assert host_module._bootstrap_aimdo(True) is False
    assert "returned False" in caplog.text
    assert "continues without successful aimdo bootstrap" in caplog.text


def test_aimdo_bootstrap_init_exception_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def fail_init() -> bool:
        raise RuntimeError("ctypes binding failed")

    control = SimpleNamespace(init=fail_init)
    monkeypatch.setattr(importlib, "import_module", lambda name: control)
    assert host_module._bootstrap_aimdo(True) is False
    assert "init() raised" in caplog.text
    assert "continues without successful aimdo bootstrap" in caplog.text


def test_aimdo_bootstrap_true_result_logs_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls: list[dict[str, int]] = []

    def init(**kwargs: int) -> bool:
        calls.append(dict(kwargs))
        return True

    control = SimpleNamespace(init=init)
    monkeypatch.setattr(importlib, "import_module", lambda name: control)
    caplog.set_level(logging.INFO, logger="dinkster.workers.aimdo")
    assert host_module._bootstrap_aimdo(True) is True
    assert calls == [{}]
    assert "completed successfully" in caplog.text


@pytest.mark.parametrize("headroom", [0, 128 * 1024**2])
def test_aimdo_bootstrap_forwards_simple_headroom(
    headroom: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls: list[dict[str, int]] = []

    def init(**kwargs: int) -> bool:
        calls.append(dict(kwargs))
        return True

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(init=init),
    )
    assert host_module._bootstrap_aimdo(True, simple_vram_headroom=headroom)
    assert calls == [{"simple_vram_headroom": headroom}]


def test_aimdo_bootstrap_old_aimdo_degrades_headroomless_and_disarms(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    caplog.set_level(logging.INFO, logger="dinkster.workers.aimdo")
    calls: list[dict[str, int]] = []

    def init(**kwargs: int) -> bool:
        calls.append(dict(kwargs))
        if kwargs:
            raise TypeError("init() got an unexpected keyword argument 'simple_vram_headroom'")
        return True

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(init=init),
    )
    assert host_module._bootstrap_aimdo(True, simple_vram_headroom=128 * 1024**2)
    assert calls == [{"simple_vram_headroom": 128 * 1024**2}, {}]
    assert "NOT applied" in caplog.text
    assert "older than 0.4.10" in caplog.text
    assert "upgrade dinkster-aimdo" in caplog.text
    assert "completed successfully with simple_vram_headroom" not in caplog.text
    assert host_module._aimdo_bootstrap_headroom_base is None


def test_aimdo_bootstrap_unrelated_typeerror_does_not_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls: list[dict[str, int]] = []

    def init(**kwargs: int) -> bool:
        calls.append(dict(kwargs))
        raise TypeError("unrelated failure inside init")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(init=init),
    )
    assert host_module._bootstrap_aimdo(True, simple_vram_headroom=64 * 1024**2) is False
    assert calls == [{"simple_vram_headroom": 64 * 1024**2}]
    assert "init() raised" in caplog.text
    assert "continues without successful aimdo bootstrap" in caplog.text


def test_aimdo_bootstrap_after_torch_warns_without_touching_control(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    def unexpected_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    assert host_module._bootstrap_aimdo(True) is False
    assert "requested too late" in caplog.text
    assert "continues without successful aimdo bootstrap" in caplog.text


@pytest.mark.parametrize("armed", [False, True])
def test_live_aimdo_probe_admits_visible_devices(
    monkeypatch: pytest.MonkeyPatch, armed: bool
) -> None:
    from aimdo_live_nodes import AimdoArmProbe

    calls: list[str] = []
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "on" if armed else "off")
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference_torch.aimdo_activation",
        SimpleNamespace(ensure_visible_aimdo_devices=lambda: calls.append("visible") or True),
    )
    assert AimdoArmProbe.execute() == {"ready": int(armed)}
    assert calls == (["visible"] if armed else [])


@pytest.mark.parametrize("loaded", [False, True])
def test_aimdo_headroom_probe_requires_library(
    monkeypatch: pytest.MonkeyPatch, loaded: bool
) -> None:
    import ctypes

    from aimdo_live_nodes import AimdoHeadroomProbe

    library = object() if loaded else None
    calls: list[tuple[object, str]] = []

    def in_dll(dll: object, name: str) -> SimpleNamespace:
        calls.append((dll, name))
        return SimpleNamespace(value=1234)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda value: value))
    monkeypatch.setitem(
        sys.modules, "dinkster_aimdo", SimpleNamespace(control=SimpleNamespace(lib=library))
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_compat_comfy.native_arm",
        SimpleNamespace(_aimdo_mechanism_factory=lambda *_: (object(), None)),
    )
    monkeypatch.setattr(ctypes, "c_int64", SimpleNamespace(in_dll=in_dll))
    monkeypatch.setenv("DINKSTER_AIMDO_HEADROOM_TARGET", "0")
    if loaded:
        assert AimdoHeadroomProbe.execute(7) == {"native_headroom": 1234, "pending": 1}
        assert calls == [(library, "simple_vram_headroom")]
    else:
        with pytest.raises(RuntimeError, match="aimdo native library did not initialize"):
            AimdoHeadroomProbe.execute(7)
        assert calls == []


def test_accelerator_runtime_preparation_is_optional_and_calls_runtime() -> None:
    calls: list[str] = []
    runtime = SimpleNamespace(
        prepare_fp8_matmul_runtime=lambda: calls.append("prepare"),
    )

    assert not host_module._prepare_accelerator_runtime(  # pyright: ignore[reportPrivateUsage]
        False,
        lambda name: calls.append(name) or runtime,
    )
    assert host_module._prepare_accelerator_runtime(  # pyright: ignore[reportPrivateUsage]
        True,
        lambda name: calls.append(name) or runtime,
    )
    assert calls == ["dinkster_inference_torch", "prepare"]


def test_accelerator_runtime_preparation_failure_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_import(_name: str) -> object:
        raise ImportError("missing runtime")

    assert not host_module._prepare_accelerator_runtime(  # pyright: ignore[reportPrivateUsage]
        True, fail_import
    )
    assert "accelerator runtime preparation failed" in caplog.text


def test_aimdo_headroom_handler_replaces_base_and_converges_in_either_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def setter(target: int) -> bool:
        calls.append(target)
        return True

    monkeypatch.setattr(host_module, "_accelerator_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_bootstrap_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_headroom_extra_bytes", 0)
    monkeypatch.delenv("DINKSTER_AIMDO_HEADROOM_TARGET", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(set_simple_vram_headroom=setter),
    )

    # An old parent frame keeps the current base.
    host_module._handle_aimdo_headroom(40)
    assert calls == [DEFAULT_INFERENCE_RESERVE_BYTES + 140]
    assert host_module._aimdo_bootstrap_headroom_base == DEFAULT_INFERENCE_RESERVE_BYTES + 100

    # Base then extras and extras then base converge on the same target.
    host_module._handle_aimdo_headroom(40, 200)
    host_module._handle_aimdo_headroom(60)
    assert calls[-1] == DEFAULT_INFERENCE_RESERVE_BYTES + 260
    host_module._handle_aimdo_headroom(40)
    host_module._handle_aimdo_headroom(60, 200)
    assert calls[-1] == DEFAULT_INFERENCE_RESERVE_BYTES + 260
    assert host_module._accelerator_headroom_base == 200
    assert host_module._aimdo_bootstrap_headroom_base == DEFAULT_INFERENCE_RESERVE_BYTES + 200
    assert host_module._aimdo_headroom_extra_bytes == 60

    host_module._handle_aimdo_headroom(0)
    assert calls[-1] == DEFAULT_INFERENCE_RESERVE_BYTES + 200
    assert "DINKSTER_AIMDO_HEADROOM_TARGET" not in os.environ

    monkeypatch.setattr(host_module, "_aimdo_bootstrap_headroom_base", None)
    host_module._handle_aimdo_headroom(50, 300)
    assert calls[-1] == DEFAULT_INFERENCE_RESERVE_BYTES + 200


@pytest.mark.parametrize("base", [True, -1, "100", None])
def test_aimdo_headroom_handler_rejects_invalid_base_without_state_change(
    base: object,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(host_module, "_accelerator_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_bootstrap_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_headroom_extra_bytes", 25)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            set_simple_vram_headroom=lambda target: calls.append(target) or True
        ),
    )

    host_module._handle_aimdo_headroom(50, base)

    assert calls == []
    assert host_module._accelerator_headroom_base == 100
    assert host_module._aimdo_bootstrap_headroom_base == 100
    assert host_module._aimdo_headroom_extra_bytes == 25
    assert "aimdoHeadroom baseBytes" in caplog.text


@pytest.mark.parametrize("extra", [True, -1, "50", None])
def test_aimdo_headroom_handler_rejects_invalid_extra_without_state_change(
    extra: object,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(host_module, "_accelerator_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_bootstrap_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_headroom_extra_bytes", 25)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            set_simple_vram_headroom=lambda target: calls.append(target) or True
        ),
    )

    host_module._handle_aimdo_headroom(extra, 200)

    assert calls == []
    assert host_module._accelerator_headroom_base == 100
    assert host_module._aimdo_bootstrap_headroom_base == 100
    assert host_module._aimdo_headroom_extra_bytes == 25
    assert "aimdoHeadroom extraBytes" in caplog.text


def test_aimdo_base_frame_is_applied_once_after_verified_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    outcomes = iter((False, True))

    def setter(target: int) -> bool:
        calls.append(target)
        return next(outcomes)

    arm = importlib.import_module("dinkster_compat_comfy.native_arm")
    monkeypatch.setattr(host_module, "_accelerator_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_bootstrap_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_headroom_extra_bytes", 0)
    monkeypatch.delenv("DINKSTER_AIMDO_HEADROOM_TARGET", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(set_simple_vram_headroom=setter),
    )
    host_module._handle_aimdo_headroom(40, 200)
    target = DEFAULT_INFERENCE_RESERVE_BYTES + 240
    assert calls == [target]
    assert os.environ["DINKSTER_AIMDO_HEADROOM_TARGET"] == str(target)

    monkeypatch.setattr(
        arm.importlib,
        "import_module",
        lambda _name: SimpleNamespace(set_simple_vram_headroom=setter),
    )
    arm._apply_pending_aimdo_headroom()
    arm._apply_pending_aimdo_headroom()
    assert calls == [target, target]
    assert "DINKSTER_AIMDO_HEADROOM_TARGET" not in os.environ


def test_classic_headroom_handler_updates_fresh_policy_without_aimdo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_module, "_accelerator_headroom_base", 100)
    monkeypatch.setattr(host_module, "_aimdo_bootstrap_headroom_base", None)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("aimdo setter called")),
    )

    host_module._handle_aimdo_headroom(50, 200)

    assert host_module._accelerator_headroom_base == 200
    assert os.environ["DINKSTER_ACCELERATOR_HEADROOM_BYTES"] == "200"


class CapturingLauncher(Launcher):
    def __init__(self) -> None:
        self.spec: LaunchSpec | None = None

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        self.spec = spec
        raise RuntimeError("launch captured")


@pytest.mark.parametrize("grouped", [False, True])
def test_isolated_workers_default_to_aimdo_auto(tmp_path: Path, grouped: bool) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        launcher = CapturingLauncher()
        worker = (
            GroupIsolatedWorker(
                "default-group",
                (manifest,),
                core_registry(),
                launcher=launcher,
            )
            if grouped
            else IsolatedWorker(manifest, core_registry(), launcher=launcher)
        )

        with pytest.raises(RuntimeError, match="launch captured"):
            await worker.start()

        assert launcher.spec is not None
        assert launcher.spec.command[-3:] == ("--aimdo-init", "--aimdo-arm", "auto")

    asyncio.run(scenario())


def test_isolated_worker_aimdo_argv_matrix(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        for arm in ("off", "auto", "on"):
            for aimdo_init in (False, True):
                launcher = CapturingLauncher()
                worker = IsolatedWorker(
                    manifest,
                    core_registry(),
                    launcher=launcher,
                    aimdo_init=aimdo_init,
                    aimdo_arm=arm,
                )
                with pytest.raises(RuntimeError, match="launch captured"):
                    await worker.start()
                assert launcher.spec is not None
                command = launcher.spec.command
                expected = (
                    sys.executable,
                    "-m",
                    "dinkster_workers.host",
                    "--endpoint",
                    command[4],
                    "--manifest",
                    str(manifest),
                    "--shm-threshold",
                    str(DEFAULT_SHM_THRESHOLD),
                    "--comfy-args-json",
                    "[]",
                )
                if aimdo_init or arm in ("auto", "on"):
                    expected += ("--aimdo-init",)
                expected += ("--aimdo-arm", arm)
                assert command == expected

    asyncio.run(scenario())


def test_group_isolated_worker_aimdo_argv_matrix(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifest = write_iso_manifest(tmp_path)
        for arm in ("off", "auto", "on"):
            for aimdo_init in (False, True):
                launcher = CapturingLauncher()
                group = GroupIsolatedWorker(
                    "argv-group",
                    (manifest,),
                    core_registry(),
                    launcher=launcher,
                    aimdo_init=aimdo_init,
                    aimdo_arm=arm,
                )
                with pytest.raises(RuntimeError, match="launch captured"):
                    await group.start()
                assert launcher.spec is not None
                command = launcher.spec.command
                expected = (
                    sys.executable,
                    "-m",
                    "dinkster_workers.host",
                    "--endpoint",
                    command[4],
                    "--manifest",
                    str(manifest),
                    "--shm-threshold",
                    str(DEFAULT_SHM_THRESHOLD),
                    "--comfy-args-json",
                    "[]",
                )
                if aimdo_init or arm in ("auto", "on"):
                    expected += ("--aimdo-init",)
                expected += ("--aimdo-arm", arm)
                assert command == expected

    asyncio.run(scenario())


def test_isolated_worker_comfy_args_reach_child_argv_exactly_and_not_env(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        launcher = CapturingLauncher()
        supplied = ("--preview-method", "auto", "--preview-size=321")
        worker = IsolatedWorker(
            write_iso_manifest(tmp_path),
            core_registry(),
            launcher=launcher,
            comfy_args=supplied,
        )
        with pytest.raises(RuntimeError, match="launch captured"):
            await worker.start()
        assert launcher.spec is not None
        index = launcher.spec.command.index("--comfy-args-json")
        assert launcher.spec.command[index + 1] == json.dumps(supplied)
        assert all("COMFY_ARG" not in name for name in launcher.spec.env)

    asyncio.run(scenario())


def test_isolated_worker_aimdo_headroom_argv_translates_device_map(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        launcher = CapturingLauncher()
        worker = IsolatedWorker(
            write_iso_manifest(tmp_path),
            core_registry(),
            launcher=launcher,
            aimdo_arm="on",
            reserve_vram=128,
            vram_budgets={"vram:cuda:1": 20_000},
            device_map=DeviceMap({"cuda:0": "cuda:1"}),
        )
        with pytest.raises(RuntimeError, match="launch captured"):
            await worker.start()
        assert launcher.spec is not None
        assert launcher.spec.command[-6:] == (
            "--aimdo-arm",
            "on",
            "--reserve-vram",
            "128",
            "--vram-budget",
            "0=20000",
        )

    asyncio.run(scenario())


def test_isolated_worker_classic_policy_argv_includes_headroom_and_budget(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        launcher = CapturingLauncher()
        worker = IsolatedWorker(
            write_iso_manifest(tmp_path),
            core_registry(),
            launcher=launcher,
            aimdo_arm="off",
            reserve_vram=128,
            vram_budgets={"vram:cuda:0": 20_000},
        )
        with pytest.raises(RuntimeError, match="launch captured"):
            await worker.start()
        assert launcher.spec is not None
        assert launcher.spec.command[-4:] == (
            "--reserve-vram",
            "128",
            "--vram-budget",
            "0=20000",
        )

    asyncio.run(scenario())


def test_isolated_worker_unmappable_budget_degrades_with_diagnostic(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    IsolatedWorker(
        write_iso_manifest(tmp_path),
        core_registry(),
        aimdo_arm="on",
        vram_budgets={"vram:cuda:0": 20_000},
        device_map=DeviceMap({"cuda:0": "cuda:1"}),
    )
    assert "accelerator budget namespace translation unavailable" in caplog.text
    assert "worker receives no budget" in caplog.text


def test_isolated_worker_ambiguous_budget_inverse_degrades(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    IsolatedWorker(
        write_iso_manifest(tmp_path),
        core_registry(),
        aimdo_arm="on",
        vram_budgets={"vram:cuda:1": 20_000},
        device_map=DeviceMap({"cuda:0": "cuda:1", "cuda:2": "cuda:1"}),
    )
    assert "accelerator budget namespace translation is ambiguous" in caplog.text
    assert "worker receives no budget" in caplog.text


def test_isolated_worker_rejects_unknown_aimdo_arm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aimdo_arm"):
        IsolatedWorker(
            write_iso_manifest(tmp_path),
            core_registry(),
            aimdo_arm="sticky",
        )


class FakeHeadroomSession:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.frames: list[dict[str, object]] = []

    async def send(self, header: object, blobs: object) -> None:
        del blobs
        if self.fail:
            raise ConnectionError("dying")
        assert isinstance(header, dict)
        self.frames.append(header)


def test_headroom_mirror_translates_deduplicates_and_tracks_lifecycle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def settle() -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def scenario() -> None:
        governor = MemoryGovernor()
        mirror = HeadroomMirror(governor, base_bytes=100)
        identity = FakeHeadroomSession()
        pinned = FakeHeadroomSession()
        dying = FakeHeadroomSession(fail=True)
        await mirror.register(identity, None)
        await mirror.register(pinned, DeviceMap({"cuda:0": "cuda:1"}))
        await mirror.register(dying, None)
        assert identity.frames == [{"type": "aimdoHeadroom", "extraBytes": 0, "baseBytes": 100}]
        identity.frames.clear()
        pinned.frames.clear()

        # Non-VRAM changes are ignored, and a repeated zero is deduplicated.
        async with governor.reserve("ram", 9):
            await settle()
        async with governor.reserve("vram:cuda:0", 0):
            await settle()
        assert identity.frames == []
        assert pinned.frames == []

        # Identity sees both parent devices and takes their max. The pinned
        # worker sees only parent cuda:1, translated to child cuda:0.
        async with governor.reserve("vram:cuda:0", 20):
            await settle()
            assert identity.frames[-1]["extraBytes"] == 20
            assert pinned.frames == []
            async with governor.reserve("vram:cuda:1", 35):
                await settle()
                assert identity.frames[-1]["extraBytes"] == 35
                assert pinned.frames[-1]["extraBytes"] == 35
            await settle()
            assert identity.frames[-1]["extraBytes"] == 20
            assert pinned.frames[-1]["extraBytes"] == 0
        await settle()
        assert identity.frames[-1]["extraBytes"] == 0

        # A base change sends despite unchanged extras; the same pair dedups.
        before_count = len(identity.frames)
        mirror.set_base(200)
        await settle()
        assert identity.frames[-1] == {
            "type": "aimdoHeadroom",
            "extraBytes": 0,
            "baseBytes": 200,
        }
        assert len(identity.frames) == before_count + 1
        mirror.set_base(200)
        await settle()
        assert len(identity.frames) == before_count + 1

        # Registration always sends the tracked pair, making argv bootstrap
        # agreement explicit and idempotent for workers started after a PUT.
        late = FakeHeadroomSession()
        await mirror.register(late, None)
        assert late.frames == [{"type": "aimdoHeadroom", "extraBytes": 0, "baseBytes": 200}]

        before = list(identity.frames)
        mirror.deregister(identity)
        async with governor.reserve("vram:cuda:0", 10):
            await settle()
        assert identity.frames == before

        ambiguous = FakeHeadroomSession()
        caplog.clear()
        await mirror.register(
            ambiguous,
            DeviceMap({"cuda:0": "cuda:1", "cuda:2": "cuda:1"}),
        )
        assert ambiguous.frames[-1]["extraBytes"] == 0
        assert "translation is ambiguous for vram:cuda:1" in caplog.text

        unmappable = FakeHeadroomSession()
        caplog.clear()
        await mirror.register(unmappable, DeviceMap({"cuda:0": "cuda:1"}))
        async with governor.reserve("vram:cuda:0", 11):
            await settle()
        assert unmappable.frames[-1]["extraBytes"] == 0
        assert "translation unavailable for vram:cuda:0" in caplog.text

    asyncio.run(scenario())


def test_isolated_worker_registers_classic_and_aimdo_policy_sessions(
    tmp_path: Path,
) -> None:
    class FakeMirror:
        def __init__(self) -> None:
            self.registered: list[object] = []
            self.deregistered: list[object] = []

        async def register(self, session: object, device_map: object) -> None:
            del device_map
            self.registered.append(session)

        def deregister(self, session: object) -> None:
            self.deregistered.append(session)

    async def scenario() -> None:
        for arm in ("off", "on"):
            mirror = FakeMirror()
            (tmp_path / arm).mkdir()
            worker = iso_worker(
                tmp_path / arm,
                core_registry(),
                aimdo_arm=arm,
                reserve_vram=256,
                headroom_mirror=mirror,
            )
            await worker.start()
            assert len(mirror.registered) == 1
            await worker.close()
            assert len(mirror.deregistered) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", [None, "auto", "on", "off"])
def test_host_main_sets_aimdo_arm_only_from_argv(
    mode: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str | None, str | None, str | None]] = []
    bootstrap: list[tuple[bool, int | None]] = []

    async def fake_serve(
        endpoint: str,
        manifest_path: str,
        *,
        shm_threshold: int,
        use_shm: bool,
    ) -> None:
        del endpoint, manifest_path, shm_threshold, use_shm
        seen.append(
            (
                os.environ.get("DINKSTER_AIMDO_ARM"),
                os.environ.get("DINKSTER_ACCELERATOR_BUDGETS"),
                os.environ.get("DINKSTER_ACCELERATOR_HEADROOM_BYTES"),
            )
        )

    def fake_bootstrap(enabled: bool, *, simple_vram_headroom: int | None = None) -> bool:
        bootstrap.append((enabled, simple_vram_headroom))
        return True

    argv = [
        "dinkster_workers.host",
        "--endpoint",
        "unix:test",
        "--manifest",
        "test.toml",
    ]
    if mode is not None:
        argv.extend(("--aimdo-arm", mode))
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "on")
    monkeypatch.setenv("DINKSTER_ACCELERATOR_BUDGETS", "0=1")
    monkeypatch.setenv("DINKSTER_ACCELERATOR_HEADROOM_BYTES", "1")
    monkeypatch.setattr(host_module, "configure_logging_from_env", lambda _env: None)
    monkeypatch.setattr(host_module, "_bootstrap_aimdo", fake_bootstrap)
    monkeypatch.setattr(host_module, "serve", fake_serve)
    monkeypatch.setattr(sys, "argv", argv)

    host_module.main()

    assert seen == [
        (
            mode or "auto",
            None,
            str(256 * 1024**2) if mode != "off" else None,
        )
    ]
    assert bootstrap == [
        (
            mode != "off",
            DEFAULT_INFERENCE_RESERVE_BYTES + 256 * 1024**2 if mode != "off" else None,
        )
    ]


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("[]", ()),
        ('["--preview-size", "321"]', ("--preview-size", "321")),
    ],
)
def test_host_main_pins_pack_argv_to_supplied_comfy_args(
    supplied: str,
    expected: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, ...]] = []

    async def fake_serve(*_args: object, **_kwargs: object) -> None:
        seen.append(tuple(sys.argv[1:]))

    monkeypatch.setattr(host_module, "configure_logging_from_env", lambda _env: None)
    monkeypatch.setattr(host_module, "_bootstrap_aimdo", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(host_module, "serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:test",
            "--manifest",
            "test.toml",
            "--comfy-args-json",
            supplied,
        ],
    )

    host_module.main()

    assert seen == [expected]


def test_host_refuses_unpaired_group_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:first",
            "--endpoint",
            "unix:second",
            "--manifest",
            "first.toml",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        host_module.main()


def test_host_forwards_repeated_group_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    async def fake_serve_many(
        endpoints: tuple[str, ...] | list[str],
        manifests: tuple[str, ...] | list[str],
        **_kwargs: object,
    ) -> None:
        seen.append((tuple(endpoints), tuple(manifests)))

    monkeypatch.setattr(host_module, "configure_logging_from_env", lambda _env: None)
    monkeypatch.setattr(host_module, "_bootstrap_aimdo", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(host_module, "serve_many", fake_serve_many)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:first",
            "--manifest",
            "first.toml",
            "--endpoint",
            "unix:second",
            "--manifest",
            "second.toml",
        ],
    )
    host_module.main()
    assert seen == [(("unix:first", "unix:second"), ("first.toml", "second.toml"))]


def test_host_main_publishes_only_validated_headroom_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str | None, str | None, str | None]] = []
    bootstrap: list[tuple[bool, int | None]] = []

    async def fake_serve(
        endpoint: str,
        manifest_path: str,
        *,
        shm_threshold: int,
        use_shm: bool,
    ) -> None:
        del endpoint, manifest_path, shm_threshold, use_shm
        seen.append(
            (
                os.environ.get("DINKSTER_AIMDO_ARM"),
                os.environ.get("DINKSTER_ACCELERATOR_BUDGETS"),
                os.environ.get("DINKSTER_ACCELERATOR_HEADROOM_BYTES"),
            )
        )

    def fake_bootstrap(enabled: bool, *, simple_vram_headroom: int | None = None) -> bool:
        bootstrap.append((enabled, simple_vram_headroom))
        return True

    monkeypatch.setenv("DINKSTER_ACCELERATOR_BUDGETS", "9=9")
    monkeypatch.setenv("DINKSTER_ACCELERATOR_HEADROOM_BYTES", "9")
    monkeypatch.setattr(host_module, "configure_logging_from_env", lambda _env: None)
    monkeypatch.setattr(host_module, "_bootstrap_aimdo", fake_bootstrap)
    monkeypatch.setattr(host_module, "serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:test",
            "--manifest",
            "test.toml",
            "--aimdo-init",
            "--aimdo-arm",
            "on",
            "--reserve-vram",
            "123",
            "--vram-budget",
            "1=456",
            "--vram-budget",
            "0=789",
        ],
    )

    host_module.main()

    assert bootstrap == [(True, DEFAULT_INFERENCE_RESERVE_BYTES + 123)]
    assert seen == [("on", "0=789,1=456", "123")]


def test_host_main_publishes_classic_policy_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str | None, str | None, str | None]] = []

    async def fake_serve(
        endpoint: str,
        manifest_path: str,
        *,
        shm_threshold: int,
        use_shm: bool,
    ) -> None:
        del endpoint, manifest_path, shm_threshold, use_shm
        seen.append(
            (
                os.environ.get("DINKSTER_AIMDO_ARM"),
                os.environ.get("DINKSTER_ACCELERATOR_BUDGETS"),
                os.environ.get("DINKSTER_ACCELERATOR_HEADROOM_BYTES"),
            )
        )

    monkeypatch.setattr(host_module, "configure_logging_from_env", lambda _env: None)
    monkeypatch.setattr(host_module, "_bootstrap_aimdo", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(host_module, "serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:test",
            "--manifest",
            "test.toml",
            "--aimdo-arm",
            "off",
            "--reserve-vram",
            "123",
            "--vram-budget",
            "0=789",
        ],
    )

    host_module.main()

    assert seen == [("off", "0=789", "123")]


def test_host_main_aimdo_arm_implies_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap: list[tuple[bool, int | None]] = []

    async def fake_serve(
        endpoint: str,
        manifest_path: str,
        *,
        shm_threshold: int,
        use_shm: bool,
    ) -> None:
        del endpoint, manifest_path, shm_threshold, use_shm

    def fake_bootstrap(enabled: bool, *, simple_vram_headroom: int | None = None) -> bool:
        bootstrap.append((enabled, simple_vram_headroom))
        return True

    monkeypatch.setattr(host_module, "configure_logging_from_env", lambda _env: None)
    monkeypatch.setattr(host_module, "_bootstrap_aimdo", fake_bootstrap)
    monkeypatch.setattr(host_module, "serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:test",
            "--manifest",
            "test.toml",
            "--aimdo-arm",
            "on",
        ],
    )

    host_module.main()

    assert bootstrap == [(True, DEFAULT_INFERENCE_RESERVE_BYTES + 256 * 1024**2)]


def test_host_main_bootstraps_aimdo_before_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def fake_serve(
        endpoint: str,
        manifest_path: str,
        *,
        shm_threshold: int,
        use_shm: bool,
    ) -> None:
        order.append("serve")

    monkeypatch.setattr(
        host_module,
        "configure_logging_from_env",
        lambda env: order.append("logging"),
    )
    monkeypatch.setattr(
        host_module,
        "_bootstrap_aimdo",
        lambda enabled, **_kwargs: order.append("aimdo") or True,
    )
    monkeypatch.setattr(
        host_module,
        "_prepare_accelerator_runtime",
        lambda enabled: order.append("prepare") or True,
    )
    monkeypatch.setattr(host_module, "serve", fake_serve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster_workers.host",
            "--endpoint",
            "unix:test",
            "--manifest",
            "test.toml",
            "--aimdo-init",
        ],
    )
    host_module.main()
    assert order == ["logging", "aimdo", "prepare", "serve"]


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

    asyncio.run(scenario())


def test_isolated_subprocess_accepts_derived_fallback_and_rejects_forgery(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (tmp_path / "native_attention_nodes.py").write_text(
            "from isopack_nodes import Sleepy, register_types\n"
            "NODES = [Sleepy]\n"
            "ARM_NODES = {'native': NODES}\n"
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "native-attention-test"\nnamespaces = ["iso"]\n\n'
            '[pack.arms]\nnative = ["iso.sleepy"]\n\n'
            '[pack.entry]\nnodes = "native_attention_nodes:NODES"\n'
            'arm_nodes = "native_attention_nodes:ARM_NODES"\n'
            'types = "native_attention_nodes:register_types"\n'
        )
        registry = core_registry()
        worker = IsolatedWorker(
            manifest,
            registry,
            extra_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(TESTS_DIR / "fixtures" / "attention_provider"),
                        str(TESTS_DIR),
                    )
                )
            },
        )
        await worker.start()
        try:
            capabilities = worker.attention_capabilities
            assert isinstance(capabilities, AttentionCapabilityEvidence)
            fallback = derive_attention_route_token(
                capabilities, AttentionPolicyConfig(requested_policy="flash")
            )
            invocation = Invocation(
                invocation_id="fallback",
                node_id="fallback",
                node_type="iso.sleepy",
                inputs={
                    "value": registry.wrap("core.string", "accepted"),
                    "seconds": registry.wrap("core.float", 0.0),
                },
                effective_schema=worker.schemas["iso.sleepy"],
                attention_policy="flash",
                attention_route_token=fallback,
            )
            accepted = await worker.invoke(invocation)
            assert accepted.error is None
            assert accepted.outputs is not None
            assert accepted.outputs["value"].resolve() == "accepted"

            forged = await worker.invoke(
                replace(
                    invocation,
                    attention_route_token=replace(fallback, device_kind="forged"),
                )
            )
            assert forged.error is not None
            assert forged.error.message == (
                "attention route token does not match worker startup evidence"
            )
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_hello_advertises_legacy_conversion_and_worker_executes_resolver(
    tmp_path: Path,
) -> None:
    (tmp_path / "torch.py").write_text(
        "class Tensor: pass\ndef load(*args, **kwargs): return {}\n",
        encoding="utf-8",
    )
    safetensors = tmp_path / "safetensors"
    safetensors.mkdir()
    (safetensors / "__init__.py").write_text("", encoding="utf-8")
    (safetensors / "torch.py").write_text(
        "def save_file(*args, **kwargs): pass\n", encoding="utf-8"
    )
    source = tmp_path / "weights-without-extension"
    source.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

    async def scenario() -> None:
        worker = IsolatedWorker(
            DEV_MANIFEST,
            core_registry(),
            extra_env={"PYTHONPATH": os.pathsep.join((str(tmp_path), str(TESTS_DIR)))},
        )
        await worker.start()
        try:
            assert worker.can_convert_legacy_checkpoint is True
            assert await worker.convert_legacy_checkpoint(source, "weights.safetensors") == (
                "success",
                None,
            )
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_worker_conversion_refusal_is_distinct_from_transport_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "torch.py").write_text(
        "class Tensor: pass\n"
        "def load(*args, **kwargs): raise ValueError('unsafe checkpoint globals')\n",
        encoding="utf-8",
    )
    safetensors = tmp_path / "safetensors"
    safetensors.mkdir()
    (safetensors / "__init__.py").write_text("", encoding="utf-8")
    (safetensors / "torch.py").write_text(
        "def save_file(*args, **kwargs): pass\n", encoding="utf-8"
    )
    source = tmp_path / "unsafe.ckpt"
    source.write_bytes(b"not a weight source")

    async def scenario() -> None:
        worker = IsolatedWorker(
            DEV_MANIFEST,
            core_registry(),
            extra_env={"PYTHONPATH": os.pathsep.join((str(tmp_path), str(TESTS_DIR)))},
        )
        await worker.start()
        try:
            outcome = await worker.convert_legacy_checkpoint(source, "unsafe.ckpt")
            assert outcome is not None
            status, reason = outcome
            assert status == "refused"
            assert reason is not None and "LegacyCheckpointError" in reason
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_torch_free_worker_omits_legacy_conversion_capability(tmp_path: Path) -> None:
    async def scenario() -> None:
        worker = IsolatedWorker(DEV_MANIFEST, core_registry())
        await worker.start()
        try:
            assert worker.can_convert_legacy_checkpoint is False
            assert (
                await worker.convert_legacy_checkpoint(tmp_path / "unused", "unused.ckpt") is None
            )
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_worker_converter_delegates_retryable_classification_to_compat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusal(ValueError):
        pass

    def unavailable(_path: Path, _logical_name: str) -> None:
        raise Refusal("future-infrastructure-failure: worker runtime unavailable")

    classified: list[BaseException] = []

    def classify(error: BaseException) -> str:
        classified.append(error)
        return "retryable"

    module = SimpleNamespace(
        LegacyCheckpointError=Refusal,
        classify_conversion_error=classify,
        resolve_weight_source=unavailable,
    )
    monkeypatch.setitem(sys.modules, "dinkster_compat_comfy.legacy_sources", module)

    status, reason = host_module._convert_legacy_checkpoint(tmp_path / "asset", "asset.ckpt")

    assert status == "error"
    assert reason is not None and "future-infrastructure-failure" in reason
    assert len(classified) == 1 and isinstance(classified[0], Refusal)


def test_conversion_pending_rpc_fails_with_worker_died_on_session_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = core_registry()
    session = BoundarySession(
        registry,
        role="test worker",
        pack="test",
        codec=ValueCodec(registry),
    )
    session._schemas = {}  # noqa: SLF001 - construct a negotiated test session
    session._convert_legacy_checkpoint = True  # noqa: SLF001
    session._alive = True  # noqa: SLF001

    async def drop_frame(_header: object, _blobs: object) -> None:
        return None

    monkeypatch.setattr(session, "send", drop_frame)

    async def scenario() -> None:
        pending = asyncio.create_task(
            session.convert_legacy_checkpoint(tmp_path / "asset", "asset.ckpt")
        )
        await asyncio.sleep(0)
        await session.close()
        with pytest.raises(WorkerDied):
            await pending

    asyncio.run(scenario())


def test_graph_compile_session_request_and_correlated_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = graph_compile_session()
    sent: list[tuple[dict[str, object], list[bytes]]] = []

    async def send(header: dict[str, object], blobs: list[bytes]) -> None:
        sent.append((header, blobs))
        request_id = str(header["requestId"])
        session._graph_compile_pending[request_id].set_result(  # noqa: SLF001
            {
                "type": GRAPH_COMPILE_RESULT_TYPE,
                "requestId": request_id,
                "graph": {"nodes": {"compiled": {}}},
                "origins": {"compiled": "source"},
            }
        )

    monkeypatch.setattr(session, "send", send)

    async def scenario() -> None:
        graph = {"nodes": {"source": {"type": "test.selector"}}}
        reply = await session.compile_graph("generation-7", graph, ("source", "other"))
        assert sent == [
            (
                {
                    "type": GRAPH_COMPILE_REQUEST_TYPE,
                    "requestId": "compile-0",
                    "generationKey": "generation-7",
                    "graph": graph,
                    "targets": ["source", "other"],
                },
                [],
            )
        ]
        assert reply["requestId"] == "compile-0"
        assert reply["origins"] == {"compiled": "source"}
        assert session._graph_compile_pending == {}  # noqa: SLF001

    asyncio.run(scenario())


def test_lazy_status_session_refuses_skew_and_malformed_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = graph_compile_session()
    invocation = LazyStatusInvocation(
        request_id="lazy-1",
        node_id="consumer",
        node_type="test.lazy",
        available_inputs={},
        connected_undemanded_inputs=(),
        effective_schema=NodeSchema(node_type="test.lazy"),
        expected_execution_identity="native:dtype-test",
        diffusion_dtype="float16",
        text_dtype="float32",
        vae_dtype="bfloat16",
    )

    async def send(header: dict[str, object], _blobs: list[bytes], _segments: object) -> None:
        assert header["componentDtypes"] == {
            "diffusion": "float16",
            "textEncoder": "float32",
            "vae": "bfloat16",
        }
        request_id = str(header["requestId"])
        session._lazy_pending[request_id].set_result(  # noqa: SLF001
            (
                {
                    "type": "lazyStatusResult",
                    "requestId": request_id,
                    "requestedInputs": "not-an-array",
                },
                [],
            )
        )

    class Writer:
        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    monkeypatch.setattr(session, "_send_unlocked", send)
    monkeypatch.setattr(session, "_writer", Writer())

    async def scenario() -> None:
        skew = await session.check_lazy_status(invocation)
        assert skew.error is not None
        assert "lazy-protocol-skew" in skew.error.message

        session._lazy_status = True  # noqa: SLF001 - negotiated test session
        malformed = await session.check_lazy_status(invocation)
        assert malformed.error is not None
        assert "lazy-protocol-malformed" in malformed.error.message
        assert session._lazy_pending == {}  # noqa: SLF001

        pending: asyncio.Future[tuple[dict[str, object], list[bytes]]] = (
            asyncio.get_running_loop().create_future()
        )
        session._lazy_pending["closing"] = pending  # noqa: SLF001
        await session.close()
        assert isinstance(pending.exception(), WorkerDied)
        assert session._lazy_pending == {}  # noqa: SLF001

    asyncio.run(scenario())


def test_lazy_status_terminal_send_failure_aborts_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write_frame = write_frame

    async def fail_lazy_result(
        writer: asyncio.StreamWriter,
        header: dict[str, object],
        blobs: list[bytes],
    ) -> None:
        if header.get("type") == "lazyStatusResult":
            raise TypeError("terminal reply is not serializable")
        await real_write_frame(writer, header, blobs)

    async def scenario() -> None:
        reader, writer, server, host_task = await start_test_host()
        monkeypatch.setattr(host_module, "write_frame", fail_lazy_result)
        await write_frame(
            writer,
            {
                "type": "checkLazyStatus",
                "requestId": "terminal-failure",
                "invocationId": "terminal-failure",
                "nodeId": "lazy",
                "nodeType": "test.missing",
            },
            [],
        )
        assert await asyncio.wait_for(read_frame(reader), timeout=2) is None
        await asyncio.wait_for(host_task, timeout=2)
        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())


def test_graph_compile_named_remote_error_reply_is_transport_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = graph_compile_session()

    async def send(header: dict[str, object], _blobs: list[bytes]) -> None:
        request_id = str(header["requestId"])
        session._graph_compile_pending[request_id].set_result(  # noqa: SLF001
            {
                "type": GRAPH_COMPILE_RESULT_TYPE,
                "requestId": request_id,
                "errorName": "pack.limit-refusal",
                "error": "CompileRefusal: too many generated nodes",
            }
        )

    monkeypatch.setattr(session, "send", send)

    async def scenario() -> None:
        reply = await session.compile_graph("generation", {}, [])
        assert reply["errorName"] == "pack.limit-refusal"
        assert reply["error"] == "CompileRefusal: too many generated nodes"

    asyncio.run(scenario())


def test_graph_compile_session_cancellation_sends_cancel_and_clears_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = graph_compile_session()
    sent: list[dict[str, object]] = []

    async def send(header: dict[str, object], _blobs: list[bytes]) -> None:
        sent.append(header)

    monkeypatch.setattr(session, "send", send)

    async def scenario() -> None:
        task = asyncio.create_task(session.compile_graph("generation", {}, ("a",)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sent == [
            {
                "type": GRAPH_COMPILE_REQUEST_TYPE,
                "requestId": "compile-0",
                "generationKey": "generation",
                "graph": {},
                "targets": ["a"],
            },
            {"type": GRAPH_COMPILE_CANCEL_TYPE, "requestId": "compile-0"},
        ]
        assert session._graph_compile_pending == {}  # noqa: SLF001

    asyncio.run(scenario())


def test_graph_compile_pending_fails_on_session_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = graph_compile_session()

    async def drop_frame(_header: object, _blobs: object) -> None:
        return None

    monkeypatch.setattr(session, "send", drop_frame)

    async def scenario() -> None:
        pending = asyncio.create_task(session.compile_graph("generation", {}, ()))
        await asyncio.sleep(0)
        await session.close()
        with pytest.raises(WorkerDied):
            await pending
        assert session._graph_compile_pending == {}  # noqa: SLF001

    asyncio.run(scenario())


def test_graph_compile_pending_fails_on_peer_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = graph_compile_session()

    async def drop_frame(_header: object, _blobs: object) -> None:
        return None

    monkeypatch.setattr(session, "send", drop_frame)

    async def scenario() -> None:
        reader = asyncio.StreamReader()
        session._reader = reader  # noqa: SLF001 - drive the production death path
        session._reader_task = asyncio.create_task(session._read_loop())  # noqa: SLF001
        pending = asyncio.create_task(session.compile_graph("generation", {}, ()))
        await asyncio.sleep(0)
        reader.feed_eof()
        with pytest.raises(WorkerDied):
            await pending
        assert session._graph_compile_pending == {}  # noqa: SLF001
        await session.close()

    asyncio.run(scenario())


def test_graph_compile_host_success_preserves_correlation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, object, bool]] = []

    def compile_graph(generation: object, graph: object, targets: object, *, cancelled) -> dict:
        calls.append((generation, graph, targets, cancelled()))
        return {
            "type": "malicious-type",
            "requestId": "malicious-id",
            "graph": {"nodes": {}},
            "origins": {},
        }

    monkeypatch.setattr(
        "dinkster_workers.host.importlib.import_module",
        lambda name: SimpleNamespace(compile_inference_graph=compile_graph),
    )

    async def scenario() -> None:
        reader, writer, server, host_task = await start_test_host()
        graph = {"nodes": {"a": {"type": "pack.selector"}}}
        targets = ["a", "b"]
        await write_frame(
            writer,
            {
                "type": GRAPH_COMPILE_REQUEST_TYPE,
                "requestId": "request-9",
                "generationKey": "generation-9",
                "graph": graph,
                "targets": targets,
            },
            [],
        )
        frame = await asyncio.wait_for(read_frame(reader), 2.0)
        assert frame is not None
        assert frame[0] == {
            "type": GRAPH_COMPILE_RESULT_TYPE,
            "requestId": "request-9",
            "graph": {"nodes": {}},
            "origins": {},
            "blobs": [],
        }
        assert calls == [("generation-9", graph, targets, False)]
        await stop_test_host(writer, server, host_task)

    asyncio.run(scenario())


def test_graph_compile_isolated_subprocess_loads_runtime_fake_module(
    tmp_path: Path,
) -> None:
    (tmp_path / "fake_graph_compile.py").write_text(
        "def compile_inference_graph(generation_key, graph, targets, *, cancelled):\n"
        "    return {\n"
        "        'type': 'override',\n"
        "        'requestId': 'override',\n"
        "        'generationKeySeen': generation_key,\n"
        "        'graphSeen': graph,\n"
        "        'targetsSeen': targets,\n"
        "        'cancelledSeen': cancelled(),\n"
        "    }\n",
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        "import dinkster_inference\n"
        "from fake_graph_compile import compile_inference_graph\n"
        "dinkster_inference.compile_inference_graph = compile_inference_graph\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        worker = IsolatedWorker(
            DEV_MANIFEST,
            core_registry(),
            extra_env={"PYTHONPATH": str(tmp_path)},
        )
        await worker.start()
        try:
            graph = {"nodes": {"selector": {"type": "pack.selector"}}}
            reply = await worker.compile_graph("generation-subprocess", graph, ("selector", "tail"))
            assert reply == {
                "type": GRAPH_COMPILE_RESULT_TYPE,
                "requestId": "compile-0",
                "generationKeySeen": "generation-subprocess",
                "graphSeen": graph,
                "targetsSeen": ["selector", "tail"],
                "cancelledSeen": False,
                "blobs": [],
            }
        finally:
            await worker.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error_name", "expected"),
    [(None, GRAPH_COMPILE_ERROR_COMPILER_FAILURE), ("pack.explicit", "pack.explicit")],
)
def test_graph_compile_host_errors_preserve_explicit_name(
    error_name: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompileFailure(RuntimeError):
        pass

    def compile_graph(*_args: object, **_kwargs: object) -> dict:
        failure = CompileFailure("stable failure")
        if error_name is not None:
            failure.error_name = error_name  # type: ignore[attr-defined]
        raise failure

    monkeypatch.setattr(
        "dinkster_workers.host.importlib.import_module",
        lambda name: SimpleNamespace(compile_inference_graph=compile_graph),
    )

    async def scenario() -> None:
        reader, writer, server, host_task = await start_test_host()
        await write_frame(
            writer,
            {
                "type": GRAPH_COMPILE_REQUEST_TYPE,
                "requestId": "failure",
                "generationKey": "generation",
                "graph": {},
                "targets": [],
            },
            [],
        )
        frame = await asyncio.wait_for(read_frame(reader), 2.0)
        assert frame is not None
        assert frame[0]["errorName"] == expected
        assert frame[0]["error"] == "CompileFailure: stable failure"
        await stop_test_host(writer, server, host_task)

    asyncio.run(scenario())


def test_graph_compile_host_non_mapping_result_is_generic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_workers.host.importlib.import_module",
        lambda name: SimpleNamespace(compile_inference_graph=lambda *args, **kwargs: None),
    )

    async def scenario() -> None:
        reader, writer, server, host_task = await start_test_host()
        await write_frame(
            writer,
            {
                "type": GRAPH_COMPILE_REQUEST_TYPE,
                "requestId": "malformed",
                "generationKey": "generation",
                "graph": {},
                "targets": [],
            },
            [],
        )
        frame = await asyncio.wait_for(read_frame(reader), 2.0)
        assert frame is not None
        assert frame[0]["errorName"] == GRAPH_COMPILE_ERROR_COMPILER_FAILURE
        assert frame[0]["error"] == "TypeError: graph compiler returned a non-Mapping result"
        await stop_test_host(writer, server, host_task)

    asyncio.run(scenario())


def test_graph_compile_host_cancel_sets_callback_and_sends_no_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    saw_cancel = threading.Event()

    def observe_cancel(*args: object, **kwargs: object) -> dict:
        callback = cast("Callable[[], bool]", kwargs["cancelled"])
        started.set()
        while not callback():
            time.sleep(0.001)
        saw_cancel.set()
        return {"graph": {}, "origins": {}}

    monkeypatch.setattr(
        "dinkster_workers.host.importlib.import_module",
        lambda name: SimpleNamespace(compile_inference_graph=observe_cancel),
    )

    async def scenario() -> None:
        reader, writer, server, host_task = await start_test_host()
        await write_frame(
            writer,
            {
                "type": GRAPH_COMPILE_REQUEST_TYPE,
                "requestId": "cancel-me",
                "generationKey": "generation",
                "graph": {},
                "targets": [],
            },
            [],
        )
        assert await asyncio.to_thread(started.wait, 2.0)
        await write_frame(
            writer,
            {"type": GRAPH_COMPILE_CANCEL_TYPE, "requestId": "cancel-me"},
            [],
        )
        assert await asyncio.to_thread(saw_cancel.wait, 2.0)
        compile_tasks = [
            task
            for task in asyncio.all_tasks()
            if "run_graph_compile" in task.get_coro().__qualname__ and not task.done()
        ]
        if compile_tasks:
            _, pending = await asyncio.wait(compile_tasks, timeout=2.0)
            assert not pending
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(read_frame(reader), 0.05)
        await stop_test_host(writer, server, host_task)

    asyncio.run(scenario())


def test_graph_compile_worker_facades_forward() -> None:
    calls: list[tuple[str, object, object]] = []

    class Session:
        async def compile_graph(self, key: str, graph: object, targets: object) -> dict:
            calls.append((key, graph, targets))
            return {"requestId": key}

    session = Session()
    isolated = object.__new__(IsolatedWorker)
    isolated._session = session  # type: ignore[attr-defined]  # noqa: SLF001
    member = GroupMemberWorker(SimpleNamespace(), session)  # type: ignore[arg-type]

    async def scenario() -> None:
        graph = {"nodes": {}}
        targets = ("a", "b")
        assert await isolated.compile_graph("isolated", graph, targets) == {"requestId": "isolated"}
        assert await member.compile_graph("member", graph, targets) == {"requestId": "member"}
        assert calls == [
            ("isolated", graph, targets),
            ("member", graph, targets),
        ]

    asyncio.run(scenario())


def test_worker_instance_token_lives_and_dies_with_the_session() -> None:
    async def scenario() -> None:
        worker = IsolatedWorker(DEV_MANIFEST, core_registry())
        assert worker.instance_token is None  # nothing announced yet
        await worker.start()
        try:
            # The hello handshake announced the child's lifetime token: the
            # same identity its resident envelopes stamp as owner provenance.
            assert worker.alive
            token = worker.instance_token
            assert isinstance(token, str) and token
        finally:
            await worker.close()
        # A token names ONE live session: it must not survive the close, so
        # a resolver that forgot to pair it with `alive` still cannot match
        # a dead lifetime.
        assert not worker.alive
        assert worker.instance_token is None

    asyncio.run(scenario())


def test_isolated_run_matches_in_process_run() -> None:
    async def scenario() -> None:
        # In-process reference: pack types registered locally.
        local_registry = core_registry()
        register_scaffold_types(local_registry)
        local_engine = Engine(
            schemas=build_schemas(SCAFFOLD_NODES),
            registry=local_registry,
            worker=InProcessWorker(build_node_types(SCAFFOLD_NODES), local_registry),
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

    asyncio.run(scenario())


def test_unregistered_type_is_interrogable_but_not_resolvable() -> None:
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
            result = await engine.run(image_graph(), ["b"])
            image = result.outputs["b"]["image"]
            # The envelope is inspectable everywhere (hazard H2): meta
            # crossed the boundary even though dev.image is not registered
            # in this process.
            assert image.type_id == "dev.image"
            assert image.meta.get("shape") == [8, 16]  # JSON round trip: tuple -> list
            assert image.meta.get("dtype") == "float32"
            with pytest.raises(UnresolvablePayload, match="not registered"):
                image.resolve()
        finally:
            await worker.close()

    asyncio.run(scenario())


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

    asyncio.run(scenario())


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX signals")
def test_isolated_close_cancellation_still_kills_reaps_and_cleans(tmp_path: Path) -> None:
    async def scenario() -> None:
        worker = iso_worker(tmp_path, core_registry())
        await worker.start()
        process = worker._proc  # noqa: SLF001
        cleanup_dir = worker._tmpdir  # noqa: SLF001
        assert process is not None
        assert cleanup_dir is not None
        os.kill(process.pid, _SIGSTOP)

        close = asyncio.create_task(worker.close())
        await asyncio.sleep(0.05)
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close

        assert process.returncode == -_SIGKILL
        assert not cleanup_dir.exists()
        await worker.close()

    asyncio.run(scenario())


def test_isolated_close_is_idempotent_after_child_already_died(tmp_path: Path) -> None:
    async def scenario() -> None:
        worker = iso_worker(tmp_path, core_registry())
        await worker.start()
        process = worker._proc  # noqa: SLF001
        assert process is not None
        process.kill()
        await process.wait()

        await asyncio.gather(worker.close(), worker.close())
        assert process.returncode == _KILLED_RETURN_CODE

    asyncio.run(scenario())


def test_group_isolated_close_is_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        manifests: list[Path] = []
        for name in ("alpha", "beta"):
            directory = tmp_path / name
            directory.mkdir()
            manifest = directory / "dinkster-pack.toml"
            manifest.write_text(
                f'[pack]\nname = "{name}"\nnamespaces = ["iso"]\n\n'
                '[pack.entry]\nnodes = "isopack_nodes:NODES"\n'
                'types = "isopack_nodes:register_types"\n',
                encoding="utf-8",
            )
            manifests.append(manifest)
        worker = GroupIsolatedWorker(
            "test-group",
            manifests,
            core_registry(),
            extra_env={"PYTHONPATH": str(TESTS_DIR)},
        )
        await worker.start()
        process = worker._proc  # noqa: SLF001
        await asyncio.gather(worker.close(), worker.close())
        assert process is not None
        assert process.returncode == 0

    asyncio.run(scenario())


def test_isolated_host_supplies_export_snapshot(tmp_path: Path) -> None:
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
                prompt={"save": {"class_type": "Save", "inputs": {"seed": 4}}},
                extra_pnginfo={"workflow": {"nodes": [4, 2]}},
            )
            result = await engine.run(graph, ["submitted-node-29"], export_snapshot=snapshot)
            assert result.outputs["submitted-node-29"]["snapshot"].resolve() == (
                f"submitted-node-29|{snapshot.prompt}|{snapshot.extra_pnginfo}"
            )
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_shm_transport_for_large_payloads_and_cleanup() -> None:
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
            # ratio is a small core.float: inline either way.
            assert blend_inputs["ratio"].transport == "inline"
        finally:
            await worker.close()
        # Single-hop handoff (hazard H14): nothing left in /dev/shm.
        assert glob.glob("/dev/shm/dinkster*") == []

    asyncio.run(scenario())


def test_shm_used_above_threshold(tmp_path: Path) -> None:
    async def scenario() -> None:
        diagnostics: list[BoundaryDiagnostic] = []
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, shm_threshold=256, on_diagnostic=diagnostics.append)
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
                    "out": GraphNode("iso.blob_out", {"size": 4096}),
                    "len": GraphNode("iso.blob_len", {"blob": Link("out", "blob")}),
                }
            )
            result = await engine.run(graph, ["len"])
            assert result.outputs["len"]["length"].resolve() == 4096
            by_node = {d.node_type: d for d in diagnostics}
            blob_out = {c.edge_id: c for c in by_node["iso.blob_out"].outputs}
            assert blob_out["blob"].transport == "shm"
            assert blob_out["blob"].size_bytes > 4096
            assert blob_out["blob"].declared_codec is False  # default-codec fallback
            blob_in = {c.edge_id: c for c in by_node["iso.blob_len"].inputs}
            assert blob_in["blob"].transport == "shm"
            assert blob_in["blob"].reused is True  # relayed bytes, no re-encode
        finally:
            await worker.close()

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_isolated_lazy_hook_protocol_and_attribution(tmp_path: Path) -> None:
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
                    available_inputs={"mode": registry.wrap("core.string", mode)},
                    connected_undemanded_inputs=("value",),
                    effective_schema=schema,
                )

            cancelling = asyncio.create_task(
                worker.check_lazy_status(lazy_invocation("lazy-cancel", "awaitable-cancel"))
            )
            await asyncio.sleep(0.1)
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

    asyncio.run(scenario())


def test_isolated_lazy_route_executes_only_the_selected_dynamic_member() -> None:
    async def scenario() -> None:
        registry = core_registry()
        worker = IsolatedWorker(FOUNDATION_MANIFEST, registry)
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

    asyncio.run(scenario())


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

    asyncio.run(scenario())


def test_cancellation_reaches_the_worker(tmp_path: Path) -> None:
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
            slow = Graph(nodes={"s": GraphNode("iso.sleepy", {"value": "never", "seconds": 30.0})})
            task = asyncio.create_task(engine.run(slow, ["s"]))
            await asyncio.sleep(0.5)  # let the invocation reach the child
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # The child cancelled the node task; the worker stays healthy.
            quick = Graph(nodes={"s": GraphNode("iso.sleepy", {"value": "ok", "seconds": 0.0})})
            result = await engine.run(quick, ["s"])
            assert result.outputs["s"]["value"].resolve() == "ok"
        finally:
            await worker.close()

    asyncio.run(scenario())


def test_routing_worker_mixes_isolated_and_in_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = core_registry()
        register_scaffold_types(registry)  # local worker handles scaffold nodes

        isolated = iso_worker(tmp_path, registry)
        await isolated.start()
        try:
            local = InProcessWorker(build_node_types(SCAFFOLD_NODES), registry)
            routes = {node_type: isolated for node_type in isolated.schemas}
            worker = RoutingWorker(routes, default=local)

            schemas = {**build_schemas(SCAFFOLD_NODES), **isolated.schemas}
            engine = Engine(
                schemas=schemas, registry=registry, worker=worker, cache=MemoryLRUCache()
            )
            # A single graph spanning both placements - zero engine changes.
            graph = Graph(
                nodes={
                    "hello": GraphNode(
                        "std.string.concat", {"a": "he", "b": "llo", "separator": ""}
                    ),
                    "echo": GraphNode("iso.sleepy", {"value": Link("hello", "text")}),
                }
            )
            result = await engine.run(graph, ["echo"])
            assert result.outputs["echo"]["value"].resolve() == "hello"
        finally:
            await isolated.close()

    asyncio.run(scenario())


def test_routing_worker_unroutable_type() -> None:
    async def scenario() -> None:
        worker = RoutingWorker({})
        with pytest.raises(KeyError, match="no worker routes"):
            await worker.prepare(["nope"])

    asyncio.run(scenario())


def test_routing_worker_add_routes_is_additive_and_atomic() -> None:
    """Progressive announcement grows routes behind the Worker reference
    the engine already holds: new types become routable in place,
    re-routing an existing type is refused, and a refused delta merges
    nothing (the non-colliding half still routes afterwards)."""

    async def scenario() -> None:
        registry = core_registry()
        register_scaffold_types(registry)
        local = InProcessWorker(build_node_types(SCAFFOLD_NODES), registry)
        worker = RoutingWorker({})

        with pytest.raises(KeyError, match="no worker routes"):
            await worker.prepare(["std.string.concat"])
        worker.add_routes({"std.string.concat": local})
        await worker.prepare(["std.string.concat"])  # routes now

        with pytest.raises(ValueError, match="re-route"):
            worker.add_routes({"dev.image.gradient": local, "std.string.concat": local})
        with pytest.raises(KeyError, match="no worker routes"):
            await worker.prepare(["dev.image.gradient"])  # atomic: not merged
        worker.add_routes({"dev.image.gradient": local})
        await worker.prepare(["dev.image.gradient"])

    asyncio.run(scenario())


def test_diagnostics_report_execute_and_boundary_time(tmp_path: Path) -> None:
    async def scenario() -> None:
        diagnostics: list[BoundaryDiagnostic] = []
        registry = core_registry()
        worker = iso_worker(tmp_path, registry, on_diagnostic=diagnostics.append)
        await worker.start()
        try:
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            graph = Graph(nodes={"s": GraphNode("iso.sleepy", {"value": "hi", "seconds": 0.2})})
            await engine.run(graph, ["s"])
        finally:
            await worker.close()
        (diag,) = diagnostics
        assert diag.node_type == "iso.sleepy"
        assert diag.pack == "isopack"
        assert diag.execute_ms >= 200.0
        assert diag.round_trip_ms >= diag.execute_ms
        assert diag.boundary_ms >= 0.0
        assert {c.edge_id for c in diag.inputs} == {"value", "seconds"}
        assert {c.edge_id for c in diag.outputs} == {"value"}

    asyncio.run(scenario())


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

    asyncio.run(scenario())


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
    with pytest.raises(ManifestError, match="lowercase"):
        load_manifest(manifest('namespaces = ["Img"]\n'))
    # Separator equivalence: img_x and img.x are the SAME claim.
    with pytest.raises(ManifestError, match="same claim"):
        load_manifest(manifest('namespaces = ["img_x", "img.x"]\n'))
    with pytest.raises(ManifestError, match="overlap"):
        load_manifest(manifest('namespaces = ["img", "img.filters"]\n'))


def test_manifest_platforms(tmp_path: Path) -> None:
    """[pack] platforms: advisory OS claims in sys.platform vocabulary
    (DESIGN M8, cross-platform). Fatal when malformed - a typo'd list
    would warn on every host or none, both silently wrong."""

    def manifest(body: str) -> Path:
        path = tmp_path / "dinkster-pack.toml"
        path.write_text(f'[pack]\nname = "p"\n{body}[pack.entry]\nnodes = "m:N"\n')
        return path

    assert load_manifest(manifest("")).platforms == ()  # omitted = unrestricted
    assert load_manifest(manifest('platforms = ["linux"]\n')).platforms == ("linux",)
    # deterministic: sorted, deduplicated
    assert load_manifest(manifest('platforms = ["win32", "linux", "linux"]\n')).platforms == (
        "linux",
        "win32",
    )

    with pytest.raises(ManifestError, match="non-empty list"):
        load_manifest(manifest('platforms = "linux"\n'))
    with pytest.raises(ManifestError, match="non-empty list"):
        load_manifest(manifest("platforms = []\n"))
    with pytest.raises(ManifestError, match="non-empty list"):
        load_manifest(manifest("platforms = [3]\n"))
    # "windows" is the classic typo for win32; it must fail, not silently never match
    with pytest.raises(ManifestError, match="not a sys.platform value"):
        load_manifest(manifest('platforms = ["windows"]\n'))


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


def test_accelerator_detection_and_resolution() -> None:
    """Host accelerator selection (DESIGN M8, cross-platform): explicit
    beats env beats detection; detection is injected here because CI has
    no GPUs and must never need one."""

    def no_tool(_name: str) -> str | None:
        return None

    def no_path(_path: str) -> bool:
        return False

    assert (
        detect_accelerator(sys_platform="darwin", machine="arm64", which=no_tool, exists=no_path)
        == "mps"
    )
    assert (
        detect_accelerator(sys_platform="darwin", machine="x86_64", which=no_tool, exists=no_path)
        == "cpu"
    )
    assert (
        detect_accelerator(
            sys_platform="linux",
            which=lambda _name: None,
            exists=lambda path: path == "/proc/driver/nvidia/version",
        )
        == "cuda"
    )
    assert (
        detect_accelerator(
            sys_platform="win32",
            which=lambda name: "smi" if name == "nvidia-smi" else None,
            exists=lambda _path: False,
        )
        == "cuda"
    )
    assert (
        detect_accelerator(
            sys_platform="linux",
            which=lambda _name: None,
            exists=lambda path: path == "/sys/module/amdgpu",
        )
        == "rocm"
    )
    assert (
        detect_accelerator(
            sys_platform="linux",
            which=lambda name: "smi" if name == "xpu-smi" else None,
            exists=lambda _path: False,
        )
        == "xpu"
    )
    assert detect_accelerator(sys_platform="linux", which=no_tool, exists=no_path) == "cpu"

    # explicit selection is authoritative; env only fills in for auto
    assert resolve_accelerator("rocm", environ={"DINKSTER_ACCELERATOR": "cuda"}) == "rocm"
    assert resolve_accelerator("auto", environ={"DINKSTER_ACCELERATOR": "xpu"}) == "xpu"
    with pytest.raises(AcceleratorError, match="unknown accelerator"):
        resolve_accelerator("gpu", environ={})
    with pytest.raises(AcceleratorError, match="unknown accelerator"):
        resolve_accelerator("auto", environ={"DINKSTER_ACCELERATOR": "gpu"})


def test_runtime_detection_is_advisory_and_best_effort() -> None:
    """Runtime/toolchain facts for snapshot scope (DESIGN M8): parsed
    from the vendor's own status tool or driver files - injected here so
    CI needs no GPU - and honestly EMPTY whenever the tool is absent,
    fails, or prints something unrecognized. Never a torch import, never
    a CUDA/ROCm context."""
    smi = (
        "| NVIDIA-SMI 550.54.14              "
        "Driver Version: 550.54.14      CUDA Version: 12.4     |\n"
    )
    assert detect_runtime("cuda", run=lambda _cmd: smi, read_text=lambda _p: None) == (
        ("cuda", "12.4"),
        ("driver", "550.54.14"),
    )
    # tool missing/failing, or output unrecognized: no facts, no error
    assert detect_runtime("cuda", run=lambda _cmd: None, read_text=lambda _p: None) == ()
    assert detect_runtime("cuda", run=lambda _cmd: "garbage", read_text=lambda _p: None) == ()

    reads: list[str] = []

    def rocm_version(path: str) -> str | None:
        reads.append(path)
        return "6.2.0\n"

    assert detect_runtime("rocm", run=lambda _cmd: None, read_text=rocm_version) == (
        ("rocm", "6.2.0"),
    )
    assert reads == ["/opt/rocm/.info/version"]
    assert detect_runtime("rocm", run=lambda _cmd: None, read_text=lambda _p: None) == ()

    # no zero-context detector exists yet for these: honestly nothing
    def unexpected(_arg: object) -> None:
        raise AssertionError("cpu/mps/xpu detection must not probe anything")

    for accelerator in ("cpu", "mps", "xpu"):
        assert detect_runtime(accelerator, run=unexpected, read_text=unexpected) == ()


def test_ensure_pack_venv_selects_accelerator_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provisioning installs base requires plus the selected accelerator's
    extra-requires; snapshot pins still bypass ranges entirely. The uv
    commands are captured, not run - what matters is WHICH requirements
    reach the install command."""
    from dinkster_workers import provision

    manifest_path = tmp_path / "pack" / "dinkster-pack.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        '[pack]\nname = "accpack"\nrequires = ["numpy"]\n'
        '[pack.entry]\nnodes = "m:N"\n'
        '[pack.extra-requires]\ncuda = ["torch==2.5.1"]\n'
    )
    manifest = load_manifest(manifest_path)

    commands: list[list[str]] = []
    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", lambda command: commands.append(list(command)))

    ensure_pack_venv(manifest, venv_root=tmp_path / "v1", accelerator="cuda")
    install = commands[-1]
    assert "numpy" in install and "torch==2.5.1" in install

    ensure_pack_venv(manifest, venv_root=tmp_path / "v2", accelerator="cpu")
    install = commands[-1]
    assert "numpy" in install and "torch==2.5.1" not in install

    ensure_pack_venv(manifest, venv_root=tmp_path / "v3")  # no accelerator = base only
    install = commands[-1]
    assert "numpy" in install and "torch==2.5.1" not in install

    # snapshot pins replace ranges AND accelerator extras - exactness wins
    ensure_pack_venv(
        manifest,
        venv_root=tmp_path / "v4",
        accelerator="cuda",
        pinned=("numpy==1.26.4", "torch==2.4.0"),
    )
    install = commands[-1]
    assert "numpy==1.26.4" in install and "torch==2.4.0" in install
    assert "torch==2.5.1" not in install and "numpy" not in install


def test_pack_venv_ignores_workspace_only_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import provision

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "workspace-pack"\nversion = "1"\n'
        'dependencies = ["dinkster-api"]\n'
        "[tool.uv.sources]\ndinkster-api = { workspace = true }\n"
    )
    manifest_path = pack / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "workspace-pack"\n[pack.entry]\nnodes = "m:N"\n')
    api = tmp_path / "dinkster-api"
    api.mkdir()
    (api / "pyproject.toml").write_text(
        '[project]\nname = "dinkster-api"\nversion = "1"\ndependencies = []\n'
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", lambda command: commands.append(list(command)))

    ensure_pack_venv(
        load_manifest(manifest_path),
        venv_root=tmp_path / "venvs",
        workspace_packages=(pack, api),
    )

    install = commands[-1]
    assert install[:3] == ["uv", "pip", "install"]
    assert "--no-sources" in install
    assert install.index("--no-sources") < install.index("-e")
    assert install.count(str(pack)) == 1
    assert "dinkster-workers" in install


def test_pack_venv_reuse_requires_matching_provisioning_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import provision

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pyproject.toml").write_text(
        '[project]\nname = "test-pack"\nversion = "1"\ndependencies = []\n'
    )
    manifest_path = pack / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "input-pack"\nrequires = ["numpy"]\n[pack.entry]\nnodes = "m:N"\n'
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_project = workspace / "pyproject.toml"
    workspace_project.write_text(
        '[project]\nname = "dinkster-api"\nversion = "1"\ndependencies = ["dinkster-schema"]\n'
    )
    builds = 0

    def run(command: list[str]) -> None:
        nonlocal builds
        if command[1] != "venv":
            return
        builds += 1
        venv = Path(command[-1])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("")

    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", run)
    venv_root = tmp_path / "venvs"
    stale = venv_root / "input-pack"
    stale_python = stale / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stale_python.parent.mkdir(parents=True)
    stale_python.write_text("")
    (stale / ".dinkster-complete").write_text("complete\n")

    manifest = load_manifest(manifest_path)
    ensure_pack_venv(
        manifest,
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cpu",
    )
    marker = venv_root / manifest.name / ".dinkster-complete"
    assert marker.read_text().startswith("sha256:")
    ensure_pack_venv(
        manifest,
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cpu",
    )
    assert builds == 1

    workspace_project.write_text(
        '[project]\nname = "dinkster-api"\nversion = "1"\n'
        'dependencies = ["dinkster-schema", "dinkster-video"]\n'
    )
    ensure_pack_venv(
        manifest,
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cpu",
    )
    assert builds == 2

    workspace_project.write_text(
        workspace_project.read_text().replace("dinkster-api", "dinkster-host")
    )
    ensure_pack_venv(
        manifest,
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cpu",
    )
    assert builds == 3

    ensure_pack_venv(
        manifest,
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cuda",
    )
    assert builds == 4

    manifest_path.write_text(manifest_path.read_text().replace('"numpy"', '"numpy", "pillow"'))
    ensure_pack_venv(
        load_manifest(manifest_path),
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cuda",
    )
    assert builds == 5

    pack_project = pack / "pyproject.toml"
    pack_project.write_text(
        pack_project.read_text().replace("dependencies = []", 'dependencies = ["pillow"]')
    )
    ensure_pack_venv(
        load_manifest(manifest_path),
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cuda",
    )
    assert builds == 6

    moved = tmp_path / "moved-pack"
    shutil.copytree(pack, moved)
    ensure_pack_venv(
        load_manifest(moved / "dinkster-pack.toml"),
        venv_root=venv_root,
        workspace_packages=(workspace,),
        accelerator="cuda",
    )
    assert builds == 7


def test_ensure_group_venv_resolves_union_and_editable_installs_every_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import provision

    manifests = []
    for name, requirement in (("alpha", "numpy"), ("beta", "torch")):
        root = tmp_path / name
        root.mkdir()
        (root / "pyproject.toml").write_text("[project]\nname = 'test'\nversion = '1'\n")
        path = root / "dinkster-pack.toml"
        path.write_text(
            f'[pack]\nname = "{name}"\nrequires = ["{requirement}"]\n'
            f'[pack.entry]\nnodes = "{name}:N"\n'
        )
        manifests.append(load_manifest(path))

    commands: list[list[str]] = []
    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", lambda command: commands.append(list(command)))
    python = provision.ensure_group_venv(
        manifests, "models", venv_root=tmp_path / "venvs", accelerator="cpu"
    )
    install = commands[-1]
    expected_python = (
        tmp_path / "venvs" / "models" / "Scripts" / "python.exe"
        if os.name == "nt"
        else tmp_path / "venvs" / "models" / "bin" / "python"
    )
    assert python == expected_python
    assert "numpy" in install and "torch" in install
    assert install.count("-e") == 2
    assert str(tmp_path / "alpha") in install and str(tmp_path / "beta") in install


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


def test_ensure_pack_venv_constraints_bind_without_adding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-scope restore's portable pins ride as a --constraint file:
    ranges still decide WHAT installs, constraints only bind versions.
    Exactness (pinned) wins over constraints, and the temp constraints
    file never outlives the install command."""
    from dinkster_workers import provision

    manifest_path = tmp_path / "pack" / "dinkster-pack.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        '[pack]\nname = "conspack"\nrequires = ["numpy"]\n[pack.entry]\nnodes = "m:N"\n'
    )
    manifest = load_manifest(manifest_path)

    commands: list[list[str]] = []
    seen: dict[str, object] = {}

    def fake_run(command: list[str]) -> None:
        commands.append(list(command))
        if "--constraint" in command:
            path = Path(command[command.index("--constraint") + 1])
            seen["path"] = path
            seen["content"] = path.read_text()

    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", fake_run)

    ensure_pack_venv(
        manifest,
        venv_root=tmp_path / "v1",
        constraints=("numpy==1.26.4", "pillow==10.4.0"),
    )
    install = commands[-1]
    assert "numpy" in install  # the range still drives resolution
    assert "--constraint" in install
    assert seen["content"] == "numpy==1.26.4\npillow==10.4.0\n"
    path = seen["path"]
    assert isinstance(path, Path) and not path.exists()  # cleaned up

    # exact pins win: constraints are ignored, no --constraint at all
    ensure_pack_venv(
        manifest,
        venv_root=tmp_path / "v2",
        pinned=("numpy==1.26.4",),
        constraints=("pillow==10.4.0",),
    )
    assert "--constraint" not in commands[-1]


def test_partition_portable_is_label_driven() -> None:
    """The portable/environment-specific split reads only the pin's own
    version: a PEP 440 local label marks a vendor build that cannot
    travel. Names are never consulted - a vendor runtime wheel WITHOUT a
    label stays portable, because as a constraint it is inert unless the
    destination's resolution actually pulls it."""
    from dinkster_workers import partition_portable

    portable, environment_specific = partition_portable(
        (
            "einops==0.8.0",
            "numpy==1.26.4",
            "nvidia-cublas-cu12==12.4.5.8",
            "torch==2.5.1+cu124",
            "torchvision==0.20.1+rocm6.2",
        )
    )
    assert portable == ("einops==0.8.0", "numpy==1.26.4", "nvidia-cublas-cu12==12.4.5.8")
    assert environment_specific == ("torch==2.5.1+cu124", "torchvision==0.20.1+rocm6.2")
    assert partition_portable(()) == ((), ())


def test_hash_annotations_never_change_the_partition() -> None:
    """A pin's hash annotations are artifact identity riding along; the
    portable/environment-specific split still reads only the version in
    the bare 'name==version' half."""
    from dinkster_workers import bare_pin, partition_portable

    assert bare_pin("einops==0.8.0 --hash=sha256:aaaa") == "einops==0.8.0"
    assert bare_pin("einops==0.8.0") == "einops==0.8.0"
    portable, environment_specific = partition_portable(
        ("einops==0.8.0 --hash=sha256:aaaa", "torch==2.5.1+cu124 --hash=sha256:bbbb")
    )
    assert portable == ("einops==0.8.0 --hash=sha256:aaaa",)
    assert environment_specific == ("torch==2.5.1+cu124 --hash=sha256:bbbb",)


def test_hash_pins_annotates_portable_and_leaves_vendor_builds_bare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hash_pins queries the index (uv pip compile --generate-hashes) for
    the PORTABLE pins only: environment-specific vendor builds stay bare,
    honestly, and never reach the compile input. The compiled file's
    continuation formatting collapses to one string per pin."""
    from dinkster_workers import provision
    from dinkster_workers.provision import hash_pins

    commands: list[list[str]] = []
    seen: dict[str, str] = {}

    def fake_run(command: list[str]) -> None:
        commands.append(list(command))
        source = Path(command[-1]).read_text()
        seen["input"] = source
        lines = [
            f"{pin} \\\n    --hash=sha256:{'a' * 4} \\\n    --hash=sha256:{'b' * 4}"
            for pin in source.splitlines()
        ]
        Path(command[command.index("-o") + 1]).write_text("\n".join(lines) + "\n")

    monkeypatch.setattr(provision, "_run", fake_run)
    annotated = hash_pins(("torch==2.5.1+cu124", "einops==0.8.0"))
    assert annotated == (
        "einops==0.8.0 --hash=sha256:aaaa --hash=sha256:bbbb",
        "torch==2.5.1+cu124",
    )
    compile_command = commands[-1]
    assert "--generate-hashes" in compile_command and "--no-deps" in compile_command
    assert seen["input"] == "einops==0.8.0\n"  # the vendor build never queried

    # all-environment-specific input: nothing to query, pins unchanged
    commands.clear()
    assert hash_pins(("torch==2.5.1+cu124",)) == ("torch==2.5.1+cu124",)
    assert commands == []

    # loud failure, never a silent bare fallback
    def failing_run(command: list[str]) -> None:
        raise provision.ProvisionError("compile failed")

    monkeypatch.setattr(provision, "_run", failing_run)
    with pytest.raises(provision.ProvisionError, match="compile failed"):
        hash_pins(("einops==0.8.0",))


def test_hash_pins_validates_the_compiled_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compile that SUCCEEDS but does not return exactly the portable
    input set - fully hashed, once each, nothing else - must fail loudly
    instead of becoming an incomplete or adulterated snapshot: uv can
    legally omit pins (no-emit-package config), emit option lines
    (index configuration), or emit unhashed requirements."""
    from dinkster_workers import provision
    from dinkster_workers.provision import hash_pins

    def compile_writing(output: str):  # noqa: ANN202
        def fake_run(command: list[str]) -> None:
            Path(command[command.index("-o") + 1]).write_text(output)

        return fake_run

    good = "einops==0.8.0 --hash=sha256:aaaa\n"
    for output, fragment in (
        ("", "omitted pins: einops==0.8.0"),
        ("einops==0.8.0\n", "unverifiable"),
        (good + "einops==0.8.0 --hash=sha256:bbbb\n", "more than once"),
        (good + "extra==1.0 --hash=sha256:cccc\n", "not one of the pins"),
        (good + "--index-url https://evil.example/simple\n", "non-pin line"),
        ("einops==0.8.0 --hash=sha256:aaaa --no-build\n", "unverifiable"),
    ):
        monkeypatch.setattr(provision, "_run", compile_writing(output))
        with pytest.raises(provision.ProvisionError, match=fragment):
            hash_pins(("einops==0.8.0",))

    # PEP 503 name folding: the compiler respelling the name still
    # matches the input identity
    monkeypatch.setattr(provision, "_run", compile_writing("Ein-Ops==0.8.0 --hash=sha256:aaaa\n"))
    assert hash_pins(("ein_ops==0.8.0",)) == ("Ein-Ops==0.8.0 --hash=sha256:aaaa",)


def test_ensure_pack_venv_hashed_pins_ride_a_requirements_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hash annotations are per-requirement options only a requirements
    file can carry: annotated pins go through a temp -r file (cleaned up
    after the install command), bare pins keep riding as CLI arguments."""
    from dinkster_workers import provision

    manifest_path = tmp_path / "pack" / "dinkster-pack.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        '[pack]\nname = "hashpack"\nrequires = ["numpy"]\n[pack.entry]\nnodes = "m:N"\n'
    )
    manifest = load_manifest(manifest_path)

    commands: list[list[str]] = []
    seen: dict[str, object] = {}

    def fake_run(command: list[str]) -> None:
        commands.append(list(command))
        if "-r" in command:
            path = Path(command[command.index("-r") + 1])
            seen["path"] = path
            seen["content"] = path.read_text()

    monkeypatch.setattr(provision, "preflight_interpreter", lambda _python: (3, 12))
    monkeypatch.setattr(provision, "_run", fake_run)

    pins = ("einops==0.8.0 --hash=sha256:aaaa --hash=sha256:bbbb", "torch==2.5.1+cu124")
    ensure_pack_venv(manifest, venv_root=tmp_path / "v1", pinned=pins)
    install = commands[-1]
    assert "-r" in install and pins[0] not in install  # never a CLI argument
    assert seen["content"] == "\n".join(pins) + "\n"
    path = seen["path"]
    assert isinstance(path, Path) and not path.exists()  # cleaned up

    # bare pins are unchanged: CLI arguments, no temp file
    ensure_pack_venv(manifest, venv_root=tmp_path / "v2", pinned=("numpy==1.26.4",))
    install = commands[-1]
    assert "numpy==1.26.4" in install and "-r" not in install


def test_manifest_presentation(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """[pack.presentation] is author-declared badge data: validated at load,
    warn-and-drop per field (a bad emoji never stops a pack), absent -> None."""
    declared = tmp_path / "declared.toml"
    declared.write_text(
        "[pack]\n"
        'name = "ade"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[pack.presentation]\n"
        'display_name = "AnimateDiff-Evolved"\n'
        'abbr = "ADE"\n'
        'mark = "\N{PERFORMING ARTS}"\n'
        'color = "#8844ff"\n',
        encoding="utf-8",
    )
    presentation = load_manifest(declared).presentation
    assert presentation is not None
    assert presentation.display_name == "AnimateDiff-Evolved"
    assert presentation.abbr == "ADE"
    assert presentation.mark == "\N{PERFORMING ARTS}"
    assert presentation.color == "#8844ff"

    plain = tmp_path / "plain.toml"
    plain.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    assert load_manifest(plain).presentation is None

    bad = tmp_path / "bad.toml"
    bad.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[pack.presentation]\n"
        'abbr = "WAY-TOO-LONG-ABBR"\n'
        'mark = "a b"\n'
        'color = "purple"\n'
    )
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(bad)
    # Every malformed field dropped with a warning; nothing valid remains.
    assert manifest.presentation is None
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "abbr" in messages and "mark" in messages and "color" in messages


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


def test_manifest_icon_is_warn_and_drop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """[pack.presentation] icon follows the presentation rule: a valid
    declaration loads with the manifest, an invalid one warns and drops
    without touching the other fields or stopping the pack."""
    from icon_bytes import png_bytes

    (tmp_path / "badge.png").write_bytes(png_bytes())
    declared = tmp_path / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[pack.presentation]\n"
        'abbr = "ADE"\n'
        'icon = "badge.png"\n'
    )
    presentation = load_manifest(declared).presentation
    assert presentation is not None and presentation.icon is not None
    assert presentation.icon.media_type == "image/png"

    (tmp_path / "bad.png").write_bytes(png_bytes(16, 16))
    declared.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[pack.presentation]\n"
        'abbr = "ADE"\n'
        'icon = "bad.png"\n'
    )
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        presentation = load_manifest(declared).presentation
    assert presentation is not None
    assert presentation.icon is None  # dropped
    assert presentation.abbr == "ADE"  # sibling fields survive
    assert "icon" in " ".join(r.getMessage() for r in caplog.records)


def test_load_pack_presentation(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The presentation-only loader: reads [pack.presentation] from a file
    that need not be a loadable pack manifest (no [pack.entry] required) -
    the badge on-ramp for unported legacy packs. Everything is advisory:
    missing file, bad TOML, or no declaration is None, never an error."""
    from dinkster_workers import load_pack_presentation

    badge_only = tmp_path / "dinkster-pack.toml"
    badge_only.write_text('[pack]\n[pack.presentation]\nabbr = "RG"\ncolor = "#00ff00"\n')
    presentation = load_pack_presentation(badge_only, pack_name="rgthree (ComfyUI)")
    assert presentation is not None
    assert presentation.display_name == "rgthree (ComfyUI)"  # fallback name
    assert presentation.abbr == "RG"
    assert presentation.color == "#00ff00"

    assert load_pack_presentation(tmp_path / "missing.toml") is None

    no_declaration = tmp_path / "plain.toml"
    no_declaration.write_text('[pack]\nname = "p"\n')
    assert load_pack_presentation(no_declaration) is None

    bad_toml = tmp_path / "broken.toml"
    bad_toml.write_text("[pack\n")
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        assert load_pack_presentation(bad_toml) is None
    assert "unreadable" in " ".join(r.getMessage() for r in caplog.records)


def test_load_pack_validates_comfy_alias_carrier_ownership_and_references(
    tmp_path: Path,
) -> None:
    def write_registry(carrier: str, target_input: str = "value") -> None:
        source = ComfyAliasSource("comfy-core", "TestNode", "comfy.TestNode", "b78cec87")
        registry = ComfyAliasRegistry(
            source_schemas=(
                ComfyAliasSourceSchema(
                    NodeSchema(
                        source.node_type,
                        inputs=(InputSpec("value", TypeExpr.concrete("core.string")),),
                    ),
                    SCHEMA_WIRE_VERSION,
                ),
            ),
            records=(
                ComfyAliasRecord(
                    id="comfy_alias:comfy-core/TestNode",
                    mapping_kind="op",
                    carrier=carrier,
                    source=source,
                    replacement=ReplacementRule(
                        from_type=source.node_type,
                        cases=(
                            ReplacementCase.build(
                                carrier,
                                inputs={target_input: MappingSource.copy("value")},
                            ),
                        ),
                    ),
                    confidence=ComfyAliasConfidence("exact", ("tests/test_isolated.py",)),
                ),
            ),
        )
        (tmp_path / "comfy-aliases.json").write_text(
            json.dumps(comfy_alias_registry_to_wire(registry)), encoding="utf-8"
        )

    manifest_path = write_iso_manifest(tmp_path, name="iso")
    write_registry("iso.sleepy")
    manifest = load_manifest(manifest_path)
    from dinkster.packs import pack_info_from_manifest

    assert pack_info_from_manifest(manifest).comfy_aliases == manifest.comfy_aliases
    host_module.load_pack(manifest)

    write_registry("other.node")
    with pytest.raises(ManifestError, match="is not owned by this pack"):
        host_module.load_pack(load_manifest(manifest_path))

    write_registry("iso.missing")
    with pytest.raises(ManifestError, match="unknown carrier"):
        host_module.load_pack(load_manifest(manifest_path))

    write_registry("iso.sleepy", target_input="missing")
    with pytest.raises(ManifestError, match="not a static input id"):
        host_module.load_pack(load_manifest(manifest_path))


def test_pack_info_from_manifest(tmp_path: Path) -> None:
    """The umbrella's canonical manifest -> /api/nodes packs-table bridge."""
    from icon_bytes import png_bytes

    from dinkster.packs import pack_info_from_manifest

    (tmp_path / "badge.png").write_bytes(png_bytes())
    declared = tmp_path / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "ade"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[pack.presentation]\n"
        'abbr = "ADE"\n'
        'icon = "badge.png"\n'
    )
    info = pack_info_from_manifest(load_manifest(declared))
    # display_name falls back to the pack name; declared fields carry over.
    assert info.display_name == "ade"
    assert info.abbr == "ADE"
    # The icon crosses as digest + media type + the digested bytes, so the
    # server serves exactly what the manifest validator hashed.
    assert info.icon is not None
    assert info.icon.digest.startswith("sha256:")
    assert info.icon.media_type == "image/png"
    assert info.icon.data == png_bytes()

    plain = tmp_path / "plain.toml"
    plain.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    plain_info = pack_info_from_manifest(load_manifest(plain))
    assert plain_info.display_name == "p"
    assert plain_info.icon is None
    assert plain_info.blueprints == ()


def test_manifest_docs_validate_and_warn_drop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from dinkster_workers import manifest as manifest_module

    from dinkster.packs import pack_info_from_manifest

    docs = tmp_path / "docs"
    node = docs / "nodes" / "p.echo"
    node.mkdir(parents=True)
    assets = docs / "assets"
    assets.mkdir()
    image = b"image"
    video = b"video"
    (assets / "preview.webp").write_bytes(image)
    (assets / "demo.mp4").write_bytes(video)
    (assets / "ignored.exe").write_bytes(b"bad")
    body = (
        "\n![Preview](assets/preview.webp)\n\n"
        '```dinkster-media\nasset = "assets/demo.mp4"\n'
        'poster = "assets/preview.webp"\ncaption = "Demo"\n```\n'
    )
    page = '+++\ntitle = "Echo"\nsummary = "Returns its input."\nschema_version = 2\n+++\n' + body
    (node / "en.md").write_text(page, newline="\n")
    (node / "zh.md").write_text(page.replace('title = "Echo"', 'title = "Echo zh"'), newline="\n")
    (node / "fr.md").write_text(page, newline="\n")
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n'
        '[pack.docs]\ndir = "docs"\ndefault_locale = "en"\n'
    )

    def loaded_pages() -> tuple[manifest_module.PackDocPage, ...]:
        loaded = load_manifest(manifest_path)
        assert loaded.docs is not None
        return loaded.docs.pages

    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(manifest_path)
    assert manifest.docs is not None
    assert [(item.id, item.locale) for item in manifest.docs.pages] == [
        ("p.echo", "en"),
        ("p.echo", "zh"),
    ]
    assert [asset.source for asset in manifest.docs.assets] == [
        "assets/demo.mp4",
        "assets/preview.webp",
    ]
    assert manifest.docs.pages[0].assets[0].digest == "sha256:" + hashlib.sha256(image).hexdigest()
    assert manifest.docs.pages[0].digest == "sha256:" + hashlib.sha256(body.encode()).hexdigest()
    assert manifest.docs.pages[0].title == "Echo"
    assert manifest.docs.pages[0].summary == "Returns its input."
    assert manifest.docs.pages[0].schema_version == 2
    assert manifest.docs.pages[0].data == body.encode()
    info = pack_info_from_manifest(manifest)
    assert info.docs is not None
    assert info.docs.pages[0].data == body.encode()
    assert "unsupported locale" in caplog.text
    assert "disallowed extension" in caplog.text

    (node / "en.md").write_text(
        page
        + "\n```text\n[unsafe](javascript:example) ![sample](assets/missing.png)\n```\n"
        + '\n````markdown\n```dinkster-media\nasset = "assets/missing.mp4"\n```\n````\n'
        + "\n`[unsafe](javascript:inline)`\n"
    )
    assert len(loaded_pages()) == 2
    (node / "en.md").write_text(page)

    # Invalid or escaped content drops alone. Lowered caps exercise limits
    # without constructing multi-megabyte fixtures.
    monkeypatch.setattr(manifest_module, "DOC_IMAGE_MAX_BYTES", 2)
    assert loaded_pages() == ()
    monkeypatch.setattr(manifest_module, "DOC_IMAGE_MAX_BYTES", 2 * 1024 * 1024)

    monkeypatch.setattr(manifest_module, "DOC_PAGE_MAX_BYTES", 32)
    caplog.clear()
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        assert load_manifest(manifest_path).docs is not None
    assert loaded_pages() == ()
    assert "cap" in caplog.text

    monkeypatch.setattr(manifest_module, "DOC_PAGE_MAX_BYTES", 256 * 1024)
    (node / "en.md").write_text(page.replace("assets/demo.mp4", "assets/ignored.exe"))
    assert loaded_pages() == ()

    (node / "en.md").write_text(page + "\n[unsafe](javascript:alert)\n")
    assert loaded_pages() == ()

    (node / "en.md").write_text(page.replace("assets/demo.mp4", "assets/preview.webp"))
    assert loaded_pages() == ()

    (node / "en.md").write_text('+++\ntitle = "Echo"\nsummary = "Unterminated"\n# Body\n')
    caplog.clear()
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        assert loaded_pages() == ()
    assert "unterminated +++ front matter" in caplog.text

    outside = tmp_path.parent / "escaped-doc.md"
    outside.write_text(page)
    (node / "en.md").unlink()
    try:
        (node / "en.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    caplog.clear()
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        escaped = load_manifest(manifest_path)
    assert escaped.docs is None
    assert escaped.docs_problem is not None and "symlink" in escaped.docs_problem
    assert "symlink" in caplog.text

    manifest_path.write_text(
        '[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n'
        '[pack.docs]\ndir = "docs"\ndefault_locale = ["en"]\n'
    )
    invalid_locale = load_manifest(manifest_path)
    assert invalid_locale.docs is None
    assert invalid_locale.docs_problem is not None


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


def test_manifest_validates_doc_example_and_node_blocks(tmp_path: Path) -> None:
    import hashlib

    docs = tmp_path / "docs" / "guides" / "getting-started"
    docs.mkdir(parents=True)
    (tmp_path / "blueprint.json").write_text("{}")
    (tmp_path / "template.json").write_text("{}")
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n'
        '[pack.docs]\ndir = "docs"\ndefault_locale = "en"\n'
        '[[pack.blueprints]]\nid = "starter"\nname = "Starter"\nfile = "blueprint.json"\n'
        '[[pack.templates]]\nid = "workflow"\nname = "Workflow"\nfile = "template.json"\n'
    )

    def load_body(body: str):
        (docs / "en.md").write_text(
            '+++\ntitle = "Getting started"\nsummary = "Learn the basics."\n+++\n' + body,
            newline="\n",
        )
        manifest = load_manifest(manifest_path)
        assert manifest.docs is not None
        return manifest.docs.pages

    body = (
        '```dinkster-example\nblueprint = "starter"\ncaption = "Insert it"\n```\n'
        '```dinkster-example\ntemplate = "workflow"\n```\n'
        '```dinkster-node\nnode = "p.echo"\n```\n'
    )
    pages = load_body(body)
    assert len(pages) == 1
    assert pages[0].node_references == ("p.echo",)
    assert pages[0].data == body.encode()
    assert pages[0].digest == "sha256:" + hashlib.sha256(body.encode()).hexdigest()

    invalid = (
        "```dinkster-example\nblueprint =\n```\n",
        '```dinkster-example\nblueprint = "starter"\nextra = true\n```\n',
        '```dinkster-example\nblueprint = "starter"\ntemplate = "workflow"\n```\n',
        '```dinkster-example\ncaption = "No target"\n```\n',
        '```dinkster-example\nblueprint = "missing"\n```\n',
        '```dinkster-example\ntemplate = "missing"\n```\n',
        '```dinkster-example\nblueprint = "bad/id"\n```\n',
        '```dinkster-example\nblueprint = "starter"\ncaption = true\n```\n',
        "```dinkster-node\nnode =\n```\n",
        "```dinkster-node\nnode = 1\n```\n",
        '```dinkster-node\nnode = "p.echo"\nextra = true\n```\n',
        '```dinkster-node\nnode = "bad/id"\n```\n',
        '```dinkster-node\nnode = "p.echo"\n',
    )
    for body in invalid:
        assert load_body(body) == ()

    inert = load_body(
        '` ```dinkster-node node = "bad/id" ``` `\n\n'
        '```python\nblock = "dinkster-node"\nnode = "bad/id"\n```\n\n'
        '````markdown\n```dinkster-example\nblueprint = "missing"\n```\n````\n'
    )
    assert len(inert) == 1
    assert inert[0].node_references == ()


def test_manifest_validates_locale_catalogs_and_preserves_bytes(tmp_path: Path) -> None:
    import hashlib

    from dinkster.packs import pack_info_from_manifest

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n'
        '[[pack.blueprints]]\nid = "starter"\nname = "Starter"\nfile = "starter.json"\n'
        '[pack.docs]\ndir = "docs"\n'
    )
    (tmp_path / "starter.json").write_text("{}")
    guide = tmp_path / "docs" / "guides" / "tour"
    guide.mkdir(parents=True)
    (guide / "en.md").write_text('+++\ntitle = "Tour"\nsummary = "Tour."\n+++\nBody\n')
    locales = tmp_path / "locales"
    locales.mkdir()
    data = (
        b'{\n  "nodes": {"p.echo": {"displayName": "Echo", '
        b'"description": "Repeats.", "inputs": {"value": {"displayName": "Value", '
        b'"doc": "Input."}}, "outputs": {"value": {"displayName": "Value"}}, '
        b'"combos": {"mode": {"strict": "Strict"}}}},\n'
        b'  "blueprints": {"starter": {"name": "Starter", "description": "Start."}},\n'
        b'  "guides": {"tour": {"title": "Tour"}},\n'
        b'  "searchTerms": {"p.echo": ["repeat"]}\n}\n'
    )
    (locales / "pt-br.json").write_bytes(data)

    manifest = load_manifest(manifest_path)

    assert manifest.locale_catalog_problems == ()
    assert len(manifest.locale_catalogs) == 1
    catalog = manifest.locale_catalogs[0]
    assert catalog.locale == "pt-br"
    assert catalog.node_references == ("p.echo",)
    assert catalog.data == data
    assert catalog.digest == "sha256:" + hashlib.sha256(data).hexdigest()
    info = pack_info_from_manifest(manifest)
    assert info.locale_catalogs[0].data == data
    assert info.locale_catalogs[0].digest == catalog.digest


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ([], "root must be an object"),
        ({"unknown": {}}, "unknown fields"),
        ({"nodes": []}, "nodes must be an object"),
        ({"nodes": {"other.echo": {"displayName": "Echo"}}}, "not owned"),
        ({"nodes": {"p.echo": {}}}, "must be a non-empty object"),
        ({"nodes": {"p.echo": {"unknown": "x"}}}, "unknown fields"),
        ({"nodes": {"p.echo": {"displayName": 1}}}, "must be a non-empty string"),
        ({"nodes": {"p.echo": {"inputs": []}}}, "inputs must be an object"),
        ({"nodes": {"p.echo": {"outputs": {"value": {}}}}}, "must translate"),
        ({"nodes": {"p.echo": {"combos": {"mode": {}}}}}, "non-empty object"),
        ({"blueprints": {"missing": {"name": "Missing"}}}, "unknown blueprint"),
        ({"guides": {"missing": {"title": "Missing"}}}, "unknown guide"),
        ({"searchTerms": {"p.echo": []}}, "must be a non-empty array"),
    ),
)
def test_manifest_drops_invalid_locale_catalog_shapes(
    tmp_path: Path, document: object, message: str
) -> None:
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    locales = tmp_path / "locales"
    locales.mkdir()
    (locales / "en.json").write_text(json.dumps(document))

    manifest = load_manifest(manifest_path)

    assert manifest.locale_catalogs == ()
    assert len(manifest.locale_catalog_problems) == 1
    assert str(Path("locales") / "en.json") in manifest.locale_catalog_problems[0]
    assert message in manifest.locale_catalog_problems[0]


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


def test_manifest_charges_malformed_catalogs_to_pack_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dinkster_workers.manifest as manifest_module

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    locales = tmp_path / "locales"
    locales.mkdir()
    first = locales / "en.json"
    second = locales / "fr.json"
    first.write_text("{")
    second.write_text("{")
    monkeypatch.setattr(manifest_module, "LOCALE_CATALOG_PACK_MAX_BYTES", 1)
    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    manifest = load_manifest(manifest_path)

    assert manifest.locale_catalogs == ()
    assert read_paths == [first]
    assert any("invalid JSON" in problem for problem in manifest.locale_catalog_problems)
    assert any("pack budget" in problem for problem in manifest.locale_catalog_problems)


def test_manifest_rechecks_locale_size_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers.manifest import LOCALE_CATALOG_MAX_BYTES

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    locales = tmp_path / "locales"
    locales.mkdir()
    catalog = locales / "en.json"
    catalog.write_text("{}")
    original_read_bytes = Path.read_bytes

    def growing_read_bytes(path: Path) -> bytes:
        if path == catalog:
            return b" " * (LOCALE_CATALOG_MAX_BYTES + 1)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", growing_read_bytes)

    manifest = load_manifest(manifest_path)

    assert manifest.locale_catalogs == ()
    assert len(manifest.locale_catalog_problems) == 1
    assert "byte cap" in manifest.locale_catalog_problems[0]


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


def test_manifest_rejects_locale_catalog_root_symlink(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\n')
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (tmp_path / "locales").symlink_to(outside, target_is_directory=True)
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


@pytest.mark.parametrize("document_kind", ["blueprint", "template"])
def test_workflow_document_size_is_checked_before_and_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document_kind: str
) -> None:
    from dinkster_workers import validate_pack_blueprint, validate_pack_template
    from dinkster_workers.manifest import BLUEPRINT_MAX_BYTES, TEMPLATE_MAX_BYTES

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n')
    if document_kind == "blueprint":
        validate = validate_pack_blueprint
        max_bytes = BLUEPRINT_MAX_BYTES
    else:
        validate = validate_pack_template
        max_bytes = TEMPLATE_MAX_BYTES

    oversized_path = tmp_path / f"oversized-{document_kind}.json"
    oversized_path.write_bytes(b"x" * (max_bytes + 1))
    entry = {"id": "bounded", "name": "Bounded", "file": oversized_path.name}
    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def record_read(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    document, problem = validate(manifest_path, entry)
    assert document is None and problem is not None and "cap" in problem
    assert oversized_path not in read_paths

    growing_path = tmp_path / f"growing-{document_kind}.json"
    growing_path.write_text("{}")
    entry["file"] = growing_path.name

    def grow_after_stat(path: Path) -> bytes:
        if path == growing_path:
            return b"x" * (max_bytes + 1)
        return record_read(path)

    monkeypatch.setattr(Path, "read_bytes", grow_after_stat)
    document, problem = validate(manifest_path, entry)
    assert document is None and problem is not None and "cap" in problem


def test_manifest_blueprints_warn_and_drop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[[pack.blueprints]] follows the presentation rule scaled to entries:
    a malformed entry warns and drops WITHOUT touching its siblings or
    stopping the pack; duplicate ids keep the first declaration; entries
    past the per-pack byte budget drop with a warning."""
    from dinkster_workers import manifest as manifest_module

    (tmp_path / "good.json").write_text('{"graphs": {}}')
    (tmp_path / "other.json").write_text('{"graphs": {"g": {}}}')
    (tmp_path / "broken.json").write_text("{nope")
    declared = tmp_path / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[[pack.blueprints]]\n"
        'id = "good"\nname = "Good"\nfile = "good.json"\n'
        "[[pack.blueprints]]\n"
        'id = "broken"\nname = "Broken"\nfile = "broken.json"\n'
        "[[pack.blueprints]]\n"
        'id = "good"\nname = "Duplicate"\nfile = "other.json"\n'
        "[[pack.blueprints]]\n"
        'id = "other"\nname = "Other"\nfile = "other.json"\n'
    )
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(declared)
    # The malformed entry and the duplicate dropped; siblings survive.
    assert [bp.id for bp in manifest.blueprints] == ["good", "other"]
    assert manifest.blueprints[0].name == "Good"  # first declaration wins
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "JSON" in messages and "duplicates" in messages

    # The per-pack budget drops entries past it, never the whole pack.
    caplog.clear()
    monkeypatch.setattr(manifest_module, "BLUEPRINT_PACK_MAX_BYTES", 20)
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(declared)
    assert [bp.id for bp in manifest.blueprints] == ["good"]
    assert "budget" in " ".join(r.getMessage() for r in caplog.records)


def test_load_pack_blueprints(tmp_path: Path) -> None:
    """The blueprints-only loader: reads [[pack.blueprints]] from a file
    that need not be a loadable pack manifest (no [pack.entry] required) -
    the starter-workflow on-ramp for unported legacy packs. Advisory
    throughout: missing file or no declaration is (), never an error."""
    from dinkster_workers import load_pack_blueprints

    (tmp_path / "starter.json").write_text('{"graphs": {}}')
    blueprint_only = tmp_path / "dinkster-pack.toml"
    blueprint_only.write_text(
        '[pack]\n[[pack.blueprints]]\nid = "starter"\nname = "Starter"\nfile = "starter.json"\n'
    )
    blueprints = load_pack_blueprints(blueprint_only)
    assert [bp.id for bp in blueprints] == ["starter"]

    assert load_pack_blueprints(tmp_path / "missing.toml") == ()
    no_declaration = tmp_path / "plain.toml"
    no_declaration.write_text('[pack]\nname = "p"\n')
    assert load_pack_blueprints(no_declaration) == ()


def test_pack_info_from_manifest_blueprints(tmp_path: Path) -> None:
    """Manifest blueprints cross the umbrella bridge verbatim: the server
    asset carries the same digest and the exact digested bytes."""
    from dinkster.packs import pack_info_from_manifest

    data = '{"graphs": {"main": {}}}'
    (tmp_path / "starter.json").write_text(data)
    declared = tmp_path / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[[pack.blueprints]]\n"
        'id = "starter"\nname = "Starter"\ndescription = "A starter"\n'
        'tags = ["demo"]\nboundary_inputs = ["core.image"]\n'
        'boundary_outputs = ["core.image", "core.mask"]\n'
        'file = "starter.json"\n'
    )
    manifest = load_manifest(declared)
    info = pack_info_from_manifest(manifest)
    assert len(info.blueprints) == 1
    asset = info.blueprints[0]
    assert asset.id == "starter"
    assert asset.name == "Starter"
    assert asset.description == "A starter"
    assert asset.tags == ("demo",)
    assert asset.boundary_inputs == ("core.image",)
    assert asset.boundary_outputs == ("core.image", "core.mask")
    assert asset.digest == manifest.blueprints[0].digest
    assert asset.data == data.encode()


def test_validate_pack_template(tmp_path: Path) -> None:
    """validate_pack_template enforces the same backend-ownable boundary
    as blueprints (id grammar, shape, containment, caps, UTF-8, JSON
    object, sha256 digest) plus the 'assets' reference list shape.
    Reference EXISTENCE is the parser's job, not the validator's - the
    validator cannot know which [[pack.assets]] declarations survived."""
    import hashlib

    from dinkster_workers import validate_pack_template
    from icon_bytes import png_bytes

    manifest_path = tmp_path / "dinkster-pack.toml"
    manifest_path.write_text('[pack]\nname = "p"\n')

    data = b'{"graphs": {"main": {}}, "meta": {"name": "Portrait"}}'
    (tmp_path / "portrait.json").write_bytes(data)
    thumbnail = png_bytes()
    (tmp_path / "portrait.png").write_bytes(thumbnail)
    entry = {
        "id": "portrait",
        "name": "Portrait",
        "description": "Starter portrait workflow",
        "tags": ["image", "portrait"],
        "family": "dinkster.sd15",
        "models": ["sd15.safetensors"],
        "assets": ["base-model", "detail-lora"],
        "file": "portrait.json",
        "thumbnail": "portrait.png",
    }
    template, problem = validate_pack_template(manifest_path, entry)
    assert problem is None and template is not None
    assert template.id == "portrait"
    assert template.name == "Portrait"
    assert template.description == "Starter portrait workflow"
    assert template.tags == ("image", "portrait")
    assert template.family == "dinkster.sd15"
    assert template.models == ("sd15.safetensors",)
    # References pass shape validation verbatim; existence is checked by
    # the parser against the pack's surviving declarations.
    assert template.assets == ("base-model", "detail-lora")
    assert template.digest == "sha256:" + hashlib.sha256(data).hexdigest()
    assert template.thumbnail is not None
    assert template.thumbnail.digest == "sha256:" + hashlib.sha256(thumbnail).hexdigest()
    assert template.thumbnail.media_type == "image/png"
    assert template.data == data

    # description, tags, and assets are optional; omission is empty.
    minimal, problem = validate_pack_template(
        manifest_path, {"id": "portrait", "name": "Portrait", "file": "portrait.json"}
    )
    assert problem is None and minimal is not None
    assert minimal.description == "" and minimal.tags == () and minimal.assets == ()
    assert minimal.family == "" and minimal.models == () and minimal.thumbnail is None

    def rejected(entry: object) -> str:
        template, problem = validate_pack_template(manifest_path, entry)
        assert template is None and problem is not None
        return problem

    assert "table" in rejected("not-a-table")
    assert "'id'" in rejected({"name": "X", "file": "portrait.json"})
    assert "lowercase" in rejected({"id": "Portrait!", "name": "X", "file": "portrait.json"})
    assert "'name'" in rejected({"id": "x", "file": "portrait.json"})
    assert "'assets'" in rejected({"id": "x", "name": "X", "file": "portrait.json", "assets": [1]})
    assert "'assets'" in rejected({"id": "x", "name": "X", "file": "portrait.json", "assets": [""]})
    assert "'family'" in rejected({"id": "x", "name": "X", "file": "portrait.json", "family": 1})
    assert "'models'" in rejected({"id": "x", "name": "X", "file": "portrait.json", "models": [""]})
    assert "'thumbnail'" in rejected(
        {"id": "x", "name": "X", "file": "portrait.json", "thumbnail": "missing.png"}
    )
    assert "'file'" in rejected({"id": "x", "name": "X"})
    # The document checks ride the shared validator - one containment
    # spot-check here; the exhaustive matrix lives in the blueprint test.
    assert "relative" in rejected({"id": "x", "name": "X", "file": "../out.json"})
    (tmp_path / "array.json").write_bytes(b"[1]")
    assert "object" in rejected({"id": "x", "name": "X", "file": "array.json"})


def test_manifest_templates_warn_and_drop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[[pack.templates]] follows the blueprint rule scaled to entries:
    malformed entries, duplicates, and budget overruns warn and drop
    individually. One additional rule: a template referencing an asset id
    absent from the pack's SURVIVING [[pack.assets]] declarations drops -
    its acquisition plan could never be constructed."""
    from dinkster_assets import digest_bytes
    from dinkster_workers import manifest as manifest_module

    (tmp_path / "good.json").write_text('{"graphs": {}}')
    (tmp_path / "needy.json").write_text('{"graphs": {"g": {}}}')
    (tmp_path / "broken.json").write_text("{nope")
    (tmp_path / "model.bin").write_bytes(b"weights")
    declared = tmp_path / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[[pack.assets]]\n"
        f'id = "model"\nname = "Model"\ndigest = "{digest_bytes(b"weights")}"\n'
        'file = "model.bin"\n'
        "[[pack.templates]]\n"
        'id = "good"\nname = "Good"\nfile = "good.json"\n'
        "[[pack.templates]]\n"
        'id = "broken"\nname = "Broken"\nfile = "broken.json"\n'
        "[[pack.templates]]\n"
        'id = "good"\nname = "Duplicate"\nfile = "needy.json"\n'
        "[[pack.templates]]\n"
        'id = "dangling"\nname = "Dangling"\nfile = "needy.json"\n'
        'assets = ["model", "no-such-asset"]\n'
        "[[pack.templates]]\n"
        'id = "needy"\nname = "Needy"\nfile = "needy.json"\nassets = ["model"]\n'
    )
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(declared)
    # Malformed, duplicate, and dangling-reference entries dropped;
    # siblings survive, including the one with a VALID asset reference.
    assert [tp.id for tp in manifest.templates] == ["good", "needy"]
    assert manifest.templates[1].assets == ("model",)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "JSON" in messages and "duplicates" in messages
    assert "no-such-asset" in messages and "undeclared" in messages

    # The per-pack budget drops entries past it, never the whole pack.
    caplog.clear()
    monkeypatch.setattr(manifest_module, "TEMPLATE_PACK_MAX_BYTES", 20)
    with caplog.at_level("WARNING", logger="dinkster.workers"):
        manifest = load_manifest(declared)
    assert [tp.id for tp in manifest.templates] == ["good"]
    assert "budget" in " ".join(r.getMessage() for r in caplog.records)


def test_load_pack_templates(tmp_path: Path) -> None:
    """The templates-only loader: reads [[pack.templates]] from a file
    that need not be a loadable pack manifest - the starter-workflow
    on-ramp for unported legacy packs. Asset references validate against
    the SAME file's [[pack.assets]] declarations. Advisory throughout:
    missing file or no declaration is (), never an error."""
    from dinkster_assets import digest_bytes
    from dinkster_workers import load_pack_templates

    (tmp_path / "starter.json").write_text('{"graphs": {}}')
    (tmp_path / "model.bin").write_bytes(b"weights")
    template_only = tmp_path / "dinkster-pack.toml"
    template_only.write_text(
        "[pack]\n"
        "[[pack.assets]]\n"
        f'id = "model"\nname = "Model"\ndigest = "{digest_bytes(b"weights")}"\n'
        'file = "model.bin"\n'
        "[[pack.templates]]\n"
        'id = "starter"\nname = "Starter"\nfile = "starter.json"\n'
        'assets = ["model"]\n'
    )
    templates = load_pack_templates(template_only, pack="comfy.legacy")
    assert [tp.id for tp in templates] == ["starter"]
    assert templates[0].assets == ("model",)

    assert load_pack_templates(tmp_path / "missing.toml", pack="p") == ()
    no_declaration = tmp_path / "plain.toml"
    no_declaration.write_text('[pack]\nname = "p"\n')
    assert load_pack_templates(no_declaration, pack="p") == ()


def test_pack_info_from_manifest_templates(tmp_path: Path) -> None:
    """Manifest templates cross the umbrella bridge verbatim: the server
    asset carries the same digest, asset references, and exact digested
    bytes."""
    from dinkster_assets import digest_bytes
    from icon_bytes import png_bytes

    from dinkster.packs import pack_info_from_manifest

    data = '{"graphs": {"main": {}}}'
    (tmp_path / "starter.json").write_text(data)
    thumbnail = png_bytes()
    (tmp_path / "starter.png").write_bytes(thumbnail)
    (tmp_path / "model.bin").write_bytes(b"weights")
    declared = tmp_path / "dinkster-pack.toml"
    declared.write_text(
        "[pack]\n"
        'name = "p"\n'
        "[pack.entry]\n"
        'nodes = "m:N"\n'
        "[[pack.assets]]\n"
        f'id = "model"\nname = "Model"\ndigest = "{digest_bytes(b"weights")}"\n'
        'file = "model.bin"\n'
        "[[pack.templates]]\n"
        'id = "starter"\nname = "Starter"\ndescription = "A starter"\n'
        'tags = ["demo"]\nfamily = "dinkster.sd15"\nmodels = ["model.bin"]\n'
        'assets = ["model"]\nfile = "starter.json"\nthumbnail = "starter.png"\n'
    )
    manifest = load_manifest(declared)
    info = pack_info_from_manifest(manifest)
    assert len(info.templates) == 1
    asset = info.templates[0]
    assert asset.id == "starter"
    assert asset.name == "Starter"
    assert asset.description == "A starter"
    assert asset.tags == ("demo",)
    assert asset.family == "dinkster.sd15"
    assert asset.models == ("model.bin",)
    assert asset.assets == ("model",)
    assert asset.thumbnail is not None
    assert asset.thumbnail.data == thumbnail
    assert asset.digest == manifest.templates[0].digest
    assert asset.data == data.encode()
