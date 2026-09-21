"""H3.1 in-process hosting mechanics and tenant-contract proof."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_memory import ModelTenantHandle, TenantRegistration
from dinkster_registry import InstallError, LockedPack, Lockfile, PlanRecord
from dinkster_server import PackInfo, create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, ManifestError
from dinkster_workers.doctor import DoctorReport, Finding

import dinkster.compose as compose_module
import dinkster.installer as installer_module
from dinkster.compose import (
    CompositionError,
    PackSpec,
    ServingComposer,
    _PackTenantRegistry,
    compose_serving,
)
from dinkster.installer import (
    Installer,
    _decode_hosting_record,
    _load_hosting_topology,
    lock_local_pack,
)
from dinkster.manager import _validated_plan_topology


def test_in_process_full_release_rejects_malformed_consumer_result() -> None:
    class Consumer:
        def footprint(self, device: str) -> int:
            del device
            return 0

        async def shed(self, pressure: object) -> int:
            del pressure
            return 0

        async def full_release(self) -> object:
            return SimpleNamespace(status="complete", error="cleanup failed")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker({}, registry, memory_consumers={"cache": cast("Any", Consumer())})
        assert await worker.full_release() == (
            {
                "consumer": "cache",
                "status": "error",
                "error": "consumer returned an invalid full release result",
            },
        )

    asyncio.run(scenario())


PINS = {"torch": "2.13.0", "dinkster-aimdo": "0.4.13"}


@pytest.fixture(autouse=True)
def component_publisher_events(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[object]]:
    events: list[object] = []
    publishers: list[object] = []

    class Publisher:
        def __init__(self) -> None:
            self.close_calls = 0
            publishers.append(self)

        def close(self) -> None:
            self.close_calls += 1

    @contextmanager
    def use_component_publisher(publisher: object) -> Generator[None, None, None]:
        events.append(("enter", publisher))
        try:
            yield
        finally:
            events.append(("exit", publisher))

    monkeypatch.setattr(compose_module, "NativeComponentPublisher", Publisher)
    monkeypatch.setitem(
        cast("dict[str, Any]", sys.modules),
        "dinkster_inference_torch",
        SimpleNamespace(use_component_publisher=use_component_publisher),
    )
    return {"events": events, "publishers": publishers}


def _lockfile() -> Lockfile:
    return Lockfile(
        packs=(
            LockedPack("alpha", "1.0.0", "blake3:" + "1" * 64, "pub", ("alpha",), "registry:x"),
            LockedPack("beta", "1.0.0", "blake3:" + "2" * 64, "pub", ("beta",), "registry:x"),
        )
    )


def test_in_process_policy_sidecar_authority_and_group_contradiction(tmp_path: Path) -> None:
    (tmp_path / "hosting.toml").write_text(
        'in-process = ["alpha"]\n[venv-groups]\ng = ["alpha", "beta"]\n'
    )
    with pytest.raises(InstallError, match="both a venv group and in-process"):
        _load_hosting_topology(tmp_path, _lockfile())

    sidecar = tmp_path / "1.hosting.json"
    sidecar.write_text(
        '{"format":"dinkster.hosting/1","inProcess":["alpha"],'
        '"runtimePins":{"dinkster-aimdo":"0.4.13",'
        '"torch":"2.13.0"},"venvGroups":{}}\n'
    )
    recorded = _decode_hosting_record(sidecar)
    (tmp_path / "hosting.toml").write_text('in-process = ["beta"]\n')
    assert recorded.in_process == ("alpha",)
    assert dict(recorded.runtime_pins) == PINS

    for legacy_name in ("format", "inProcess", "runtimePins", "venvGroups"):
        legacy = tmp_path / f"legacy-{legacy_name}.hosting.json"
        legacy.write_text(json.dumps({legacy_name: ["alpha", "beta"]}))
        assert _decode_hosting_record(legacy).groups == ((legacy_name, ("alpha", "beta")),)

    plan = PlanRecord(
        target=_lockfile(),
        in_process=("alpha",),
        runtime_pins=tuple(PINS.items()),
        generation_topology=True,
    )
    assert PlanRecord.from_record_json(plan.record_json()) == plan

    policy_installer = Installer(tmp_path / "policy-root")
    (policy_installer.root / "hosting.toml").write_text(
        '[venv-groups]\nmodels = ["alpha", "beta"]\n'
    )
    with pytest.raises(InstallError, match="topology changed"):
        _validated_plan_topology(
            policy_installer,
            _lockfile(),
            (),
            (),
            {},
        )


def _manifest(directory: Path, *, extra: str = "") -> Path:
    directory.mkdir()
    path = directory / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "isopack"\nnamespaces = ["iso"]\n'
        f"{extra}\n[pack.entry]\n"
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
    )
    return path


def _install_pack(directory: Path, name: str, *, requires: str = "") -> Path:
    directory.mkdir(parents=True)
    (directory / f"{name}_nodes.py").write_text("NODES = []\n")
    requires_line = f"requires = [{requires}]\n" if requires else ""
    (directory / "dinkster-pack.toml").write_text(
        f'[pack]\nname = "{name}"\n{requires_line}[pack.entry]\nnodes = "{name}_nodes:NODES"\n'
    )
    return directory


def _placement_manifest(directory: Path) -> Path:
    directory.mkdir()
    (directory / "h31_placement_nodes.py").write_text(
        "from dinkster_schema import InputSpec,Node,NodeSchema,OutputSpec,TypeExpr\n"
        "STRING=TypeExpr.concrete('core.string')\n"
        "class Echo(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='h31.echo',\n"
        "            inputs=(InputSpec('value',STRING),),\n"
        "            outputs=(OutputSpec('value',STRING),))\n"
        "    @classmethod\n"
        "    def execute(cls,*,value): return cls.outputs(value=value)\n"
        "NODES=[Echo]\n"
    )
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "h31-placement"\nnamespaces = ["h31"]\n[pack.entry]\n'
        'nodes = "h31_placement_nodes:NODES"\n'
    )
    return manifest


def _composer(**kwargs: object) -> ServingComposer:
    return ServingComposer(
        torch_capable=lambda: True,
        runtime_versions=lambda: PINS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_in_process_placement_is_byte_identical_for_schema_provenance_and_invocation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = _placement_manifest(tmp_path / "pack")
        info = {"h31-placement": PackInfo(display_name="Placement Proof", source="test")}
        isolated = await compose_serving(
            [PackSpec(manifest, env={"PYTHONPATH": str(manifest.parent)}, packs=info)],
            include_default_packs=False,
        )
        composer = _composer()
        in_process = composer.composition
        try:
            await composer.add_pack(
                PackSpec(manifest, packs=info, in_process=True, runtime_pins=PINS)
            )

            async def surface(composition: object) -> bytes:
                current = composition  # keep the helper's type use local
                app = create_app(
                    current.make_engine,  # type: ignore[attr-defined]
                    current.schemas,  # type: ignore[attr-defined]
                    packs=current.packs,  # type: ignore[attr-defined]
                    node_packs=current.node_packs,  # type: ignore[attr-defined]
                )
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    payload = await (await client.get("/api/nodes")).json()
                    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                finally:
                    await client.close()

            assert await surface(isolated) == await surface(in_process)
            graph = Graph(nodes={"n": GraphNode("h31.echo", {"value": "same"})})
            isolated_result = await isolated.make_engine(lambda _event: None).run(graph, ["n"])
            in_process_result = await in_process.make_engine(lambda _event: None).run(graph, ["n"])
            left = isolated_result.outputs["n"]["value"]
            right = in_process_result.outputs["n"]["value"]
            assert (left.type_id, left.resolve(), left.meta) == (
                right.type_id,
                right.resolve(),
                right.meta,
            )
            assert in_process._isolated == []

            second_root = tmp_path / "second"
            second_root.mkdir()
            (second_root / "h31_second_nodes.py").write_text(
                (manifest.parent / "h31_placement_nodes.py")
                .read_text()
                .replace("h31.echo", "h32.second")
            )
            second = second_root / "dinkster-pack.toml"
            second.write_text(
                '[pack]\nname = "h31-second"\nnamespaces = ["h32"]\n'
                '[pack.entry]\nnodes = "h31_second_nodes:NODES"\n'
            )
            await composer.add_pack(PackSpec(second, in_process=True, runtime_pins=PINS))
            assert (
                composer._records["h31-placement"].domain is composer._records["h31-second"].domain
            )
            second_graph = Graph(nodes={"n": GraphNode("h32.second", {"value": "same"})})
            second_result = await composer.composition.make_engine(lambda _event: None).run(
                second_graph, ["n"]
            )
            assert second_result.outputs["n"]["value"].resolve() == "same"

            conflicting_root = tmp_path / "conflicting"
            conflicting_root.mkdir()
            (conflicting_root / "h31_second_nodes.py").write_text(
                (second_root / "h31_second_nodes.py").read_text()
            )
            conflicting = conflicting_root / "dinkster-pack.toml"
            conflicting.write_text(
                '[pack]\nname = "h31-conflicting"\nnamespaces = ["h33"]\n'
                '[pack.entry]\nnodes = "h31_second_nodes:NODES"\n'
            )
            with pytest.raises(ManifestError, match="already loaded from outside the pack root"):
                await composer.add_pack(PackSpec(conflicting, in_process=True, runtime_pins=PINS))
        finally:
            await composer.close()
            await isolated.close()

    asyncio.run(scenario())


def test_runtime_pinned_pack_owns_component_publisher_until_pack_removal(
    tmp_path: Path,
    component_publisher_events: dict[str, list[object]],
) -> None:
    async def scenario() -> None:
        manifest = _placement_manifest(tmp_path / "publisher-pack")
        module = manifest.parent / "h31_placement_nodes.py"
        publisher_module = manifest.parent / "component_publisher_nodes.py"
        module.rename(publisher_module)
        manifest.write_text(
            manifest.read_text().replace("h31_placement_nodes", "component_publisher_nodes")
        )
        spec = PackSpec(manifest, in_process=True, runtime_pins=PINS)
        composer = _composer()
        await composer.add_pack(spec)
        record = composer._records["h31-placement"]
        publisher = record.component_publisher
        assert publisher is not None
        assert record.owns_component_publisher
        assert composer.composition._component_publishers == [publisher]
        assert component_publisher_events["publishers"] == [publisher]

        component_publisher_events["events"].clear()
        graph = Graph(nodes={"n": GraphNode("h31.echo", {"value": "same"})})
        result = await composer.composition.make_engine(lambda _event: None).run(graph, ["n"])
        assert result.outputs["n"]["value"].resolve() == "same"
        assert component_publisher_events["events"] == [
            ("enter", publisher),
            ("exit", publisher),
        ]

        discarded = composer.spawn_empty()
        await discarded.add_pack(spec)
        assert discarded._records["h31-placement"].component_publisher is publisher
        assert not discarded._records["h31-placement"].owns_component_publisher
        await discarded.close()
        assert cast("Any", publisher).close_calls == 0

        staged = composer.spawn_empty()
        await staged.add_pack(spec)
        old = composer.adopt(staged)
        await old.close()
        assert cast("Any", publisher).close_calls == 0
        assert composer._records["h31-placement"].owns_component_publisher
        assert composer.composition._component_publishers == [publisher]

        await composer.remove_pack("h31-placement")
        assert cast("Any", publisher).close_calls == 1
        assert composer.composition._component_publishers == []
        await composer.close()
        assert cast("Any", publisher).close_calls == 1

    asyncio.run(scenario())


def test_in_process_installer_skips_venv_and_exact_pin_refusal_is_not_overridable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provisioned: list[str] = []
    probed: list[Path | str] = []
    serving_python = tmp_path / "serve-python"

    def provision(manifest: object, root: Path, _spec: object) -> Path:
        name = manifest.name  # type: ignore[attr-defined]
        provisioned.append(name)
        python = root / name / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("#!fake\n")
        return python

    root = tmp_path / "root"
    installer = Installer(
        root,
        provision=provision,  # type: ignore[arg-type]
        freeze=lambda _python: (),
        serving_interpreter=serving_python,
        torch_capability_probe=lambda python: not probed.append(python),
        runtime_version_probe=lambda python: probed.append(python) or PINS,
    )
    bad = _install_pack(tmp_path / "bad", "alpha", requires='"torch==2.12.0"')
    target = Lockfile.of([lock_local_pack(bad, installer.artifacts_dir)[0]])
    (root / "hosting.toml").write_text('in-process = ["alpha"]\n')
    with pytest.raises(InstallError, match="must pin torch exactly"):
        installer.apply(target, allow_doctor_findings=True)
    assert installer.current_number() is None
    assert provisioned == []
    with pytest.raises(InstallError, match="requires exact torch and dinkster-aimdo"):
        installer.apply(
            target,
            hosting_groups=(),
            hosting_in_process=("alpha",),
            hosting_runtime_pins={},
        )

    aimdo = _install_pack(tmp_path / "aimdo", "alpha", requires='"dinkster-aimdo==0.4.13"')
    aimdo_target = Lockfile.of([lock_local_pack(aimdo, installer.artifacts_dir)[0]])
    with pytest.raises(InstallError, match="must not declare a dinkster-aimdo dependency"):
        installer.apply(aimdo_target, allow_doctor_findings=True)

    probes = iter((PINS, {**PINS, "torch": "2.14.0"}))
    drift_root = tmp_path / "drift-root"
    drift_installer = Installer(
        drift_root,
        provision=provision,  # type: ignore[arg-type]
        torch_capability_probe=lambda _python: True,
        runtime_version_probe=lambda _python: next(probes),
    )
    drift_pack = _install_pack(tmp_path / "drift", "drift")
    drift_target = Lockfile.of([lock_local_pack(drift_pack, drift_installer.artifacts_dir)[0]])
    (drift_root / "hosting.toml").write_text('in-process = ["drift"]\n')
    with pytest.raises(InstallError, match="runtime changed during staging"):
        drift_installer.apply(drift_target, allow_doctor_findings=True)
    assert drift_installer.current_number() is None

    good = _install_pack(tmp_path / "good", "beta")
    target = Lockfile.of(
        [
            lock_local_pack(_install_pack(tmp_path / "alpha", "alpha"), installer.artifacts_dir)[0],
            lock_local_pack(good, installer.artifacts_dir)[0],
        ]
    )
    (root / "hosting.toml").write_text('in-process = ["alpha"]\n')
    ordinary = DoctorReport(
        "alpha", "dinkster-pack.toml", (Finding("error", "doctor.test", "ordinary"),)
    )
    monkeypatch.setattr(installer_module, "diagnose", lambda *_args, **_kwargs: ordinary)
    installer.apply(target, allow_doctor_findings=True)
    assert provisioned == ["beta"]
    assert probed and set(probed) == {serving_python}
    assert installer.packs_for_serving()[0].in_process
    assert "WARNING: allowing doctor findings" in capsys.readouterr().out
    sidecar = _decode_hosting_record(root / "generations" / "1.hosting.json")
    assert sidecar.in_process == ("alpha",)
    assert dict(sidecar.runtime_pins) == PINS
    snapshot = installer.snapshot()
    assert snapshot.in_process == ("alpha",)
    assert dict(snapshot.runtime_pins) == PINS
    assert type(snapshot).from_record_json(snapshot.record_json()) == snapshot

    (root / "hosting.toml").unlink()
    installer.apply(target, allow_doctor_findings=True)
    assert _decode_hosting_record(root / "generations" / "2.hosting.json").in_process == ()
    monkeypatch.setattr(
        installer_module,
        "diagnose",
        lambda manifest, **_kwargs: DoctorReport("pack", str(manifest), ()),
    )
    assert installer.rollback() == 3
    assert _decode_hosting_record(root / "generations" / "3.hosting.json").in_process == ("alpha",)


def test_in_process_pack_refuses_non_torch_runtime_pin_drift_and_contradictory_spec(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = _manifest(tmp_path / "pack")
        non_torch = ServingComposer(torch_capable=lambda: False)
        with pytest.raises(CompositionError, match="torch-capable"):
            await non_torch.add_pack(PackSpec(manifest, in_process=True, runtime_pins=PINS))
        drift = ServingComposer(
            torch_capable=lambda: True,
            runtime_versions=lambda: {**PINS, "torch": "2.14.0"},
        )
        with pytest.raises(CompositionError, match="runtime drift"):
            await drift.add_pack(PackSpec(manifest, in_process=True, runtime_pins=PINS))
        with pytest.raises(CompositionError, match="cannot set"):
            drift.validate_specs(
                [PackSpec(manifest, python="python", in_process=True, runtime_pins=PINS)]
            )
        await non_torch.close()
        await drift.close()

    asyncio.run(scenario())


def test_in_process_pack_refuses_extensions_registry_absence_and_reload(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        extension = _manifest(
            tmp_path / "extension",
            extra='[pack.extension]\nschema = "isopack_nodes:NODES"\nprivileges = ["schema"]',
        )
        composer = _composer()
        with pytest.raises(CompositionError, match="nodes only"):
            await composer.add_pack(PackSpec(extension, in_process=True, runtime_pins=PINS))
        await composer.close()

        residency = _manifest(tmp_path / "residency")
        residency.write_text(
            residency.read_text().replace(
                'nodes = "isopack_nodes:NODES"',
                'nodes = "isopack_nodes:NODES"\nconsumers = "isopack_nodes:consumers"',
            )
        )
        absent = _composer()
        with pytest.raises(CompositionError, match="no model tenant registry"):
            await absent.add_pack(PackSpec(residency, in_process=True, runtime_pins=PINS))
        await absent.close()

        plain = _manifest(tmp_path / "plain")
        (plain.parent / "isopack_nodes.py").write_text(
            (Path(__file__).parent / "isopack_nodes.py").read_text()
        )
        loaded = _composer()
        spec = PackSpec(plain, in_process=True, runtime_pins=PINS)
        await loaded.add_pack(spec)
        with pytest.raises(CompositionError, match="cannot reload in-process"):
            await loaded.reload_pack("isopack")
        staged = loaded.spawn_empty()
        delta = await staged.add_pack(spec)
        assert "iso.chatty" in delta.schemas
        changed_staged = loaded.spawn_empty()
        with pytest.raises(CompositionError, match="new or changed.*restart dinkster-serve"):
            await changed_staged.add_pack(
                PackSpec(
                    plain,
                    in_process=True,
                    runtime_pins={**PINS, "torch": "2.14.0"},
                )
            )
        await changed_staged.close()
        await staged.close()
        await loaded.close()

    asyncio.run(scenario())


def test_in_process_tenant_registers_on_invocation_unregisters_on_remove_and_refusal_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "tenant_nodes.py"
    module.write_text(
        "from dinkster_memory import model_tenant_registry\n"
        "from dinkster_schema import Node,NodeSchema,OutputSpec,TypeExpr\n"
        "ALLOCATIONS=[]\n"
        "class Handle:\n"
        "    pack='tenantpack'\n"
        "    model_id='model'\n"
        "    size_estimate_bytes=4\n"
        "    device_preference='cuda:0'\n"
        "    async def offload(self): pass\n"
        "    async def evict(self): pass\n"
        "def unused(): return {}\n"
        "class TenantNode(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        output=OutputSpec('device',TypeExpr.concrete('core.string'))\n"
        "        return NodeSchema(node_type='tenant.load',outputs=(output,))\n"
        "    @classmethod\n"
        "    async def execute(cls):\n"
        "        registration=await model_tenant_registry().register(Handle())\n"
        "        await model_tenant_registry().notify_resident(registration)\n"
        "        ALLOCATIONS.append(registration.assigned_device)\n"
        "        return cls.outputs(device=registration.assigned_device)\n"
        "NODES=[TenantNode]\n"
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "tenantpack"\nnamespaces = ["tenant"]\n[pack.entry]\n'
        'nodes = "tenant_nodes:NODES"\nconsumers = "tenant_nodes:unused"\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    async def invoke(composer: ServingComposer) -> Any:
        await composer.add_pack(PackSpec(manifest, in_process=True, runtime_pins=PINS))
        graph = Graph(nodes={"n": GraphNode("tenant.load", {})})
        return await composer.composition.make_engine(lambda _event: None).run(graph, ["n"])

    async def scenario() -> None:
        registry = _StrictFakeRegistry(assigned_device="cuda:1")
        composer = _composer(tenant_registry=registry)
        result = await invoke(composer)
        assert result.outputs["n"]["device"].resolve() == "cuda:1"
        loaded = importlib.import_module("tenant_nodes")
        assert loaded.ALLOCATIONS == ["cuda:1"]
        assert tuple(registry.live) == (("tenantpack", "model"),)
        discarded = composer.spawn_empty()
        await discarded.add_pack(PackSpec(manifest, in_process=True, runtime_pins=PINS))
        assert discarded._records["tenantpack"].worker is not composer._records["tenantpack"].worker
        await discarded.close()
        assert tuple(registry.live) == (("tenantpack", "model"),)
        staged = composer.spawn_empty()
        await staged.add_pack(PackSpec(manifest, in_process=True, runtime_pins=PINS))
        old = composer.adopt(staged)
        await old.close()
        assert tuple(registry.live) == (("tenantpack", "model"),)
        workers = await composer.full_free("in-process-tenants")
        tenant_worker = next(worker for worker in workers if worker["worker"] == "tenantpack")
        assert tenant_worker["status"] == "complete"
        assert tenant_worker["consumers"] == [{"consumer": "model-tenants", "status": "complete"}]
        assert registry.live == {}
        await composer.remove_pack("tenantpack")
        assert registry.live == {}
        assert len(registry.notifications) == 1
        await composer.close()

        refused_registry = _StrictFakeRegistry(budget=0)
        refused = _composer(tenant_registry=refused_registry)
        with pytest.raises(ExecutionError, match="tenant budget refused"):
            await invoke(refused)
        assert loaded.ALLOCATIONS == ["cuda:1"]
        assert refused_registry.live == {}
        await refused.close()

    asyncio.run(scenario())


@dataclass
class _Handle:
    pack: str
    model_id: str
    size_estimate_bytes: int
    device_preference: str | None = None
    registry: Any | None = None
    started: asyncio.Event | None = None
    release: asyncio.Event | None = None
    offloads: int = 0
    evictions: int = 0

    async def offload(self) -> None:
        self.offloads += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.registry is not None:
            await self.registry.register(self)

    async def evict(self) -> None:
        self.evictions += 1


class _StrictFakeRegistry:
    """Executable specification for every H3.1 tenant amendment."""

    def __init__(self, budget: int = 1_000, assigned_device: str = "cuda:1") -> None:
        self.budget = budget
        self.assigned_device = assigned_device
        self.live: dict[tuple[str, str], tuple[ModelTenantHandle, TenantRegistration, int]] = {}
        self.notifications: list[TenantRegistration] = []
        self._callback_owner: asyncio.Task[object] | None = None
        self._lock = asyncio.Lock()

    async def register(self, handle: ModelTenantHandle) -> TenantRegistration:
        if asyncio.current_task() is self._callback_owner:
            raise RuntimeError("tenant callbacks are non-reentrant")
        async with self._lock:
            key = (handle.pack, handle.model_id)
            if key in self.live:
                raise RuntimeError("duplicate tenant identity")
            used = sum(item[2] for item in self.live.values())
            if used + handle.size_estimate_bytes > self.budget:
                raise RuntimeError("tenant budget refused")
            registration = TenantRegistration(
                f"{handle.pack}:{handle.model_id}", self.assigned_device
            )
            self.live[key] = (handle, registration, handle.size_estimate_bytes)
            return registration

    async def unregister(self, registration: TenantRegistration) -> None:
        if asyncio.current_task() is self._callback_owner:
            raise RuntimeError("tenant callbacks are non-reentrant")
        async with self._lock:
            key = next(key for key, item in self.live.items() if item[1] == registration)
            del self.live[key]

    async def terminal_release(self, registration: TenantRegistration) -> None:
        await self.evict(registration)
        await self.unregister(registration)

    async def notify_resident(self, registration: TenantRegistration) -> None:
        if asyncio.current_task() is self._callback_owner:
            raise RuntimeError("tenant callbacks are non-reentrant")
        async with self._lock:
            item = next(item for item in self.live.values() if item[1] == registration)
            if item[2] > self.budget:
                raise RuntimeError("tenant budget refused")
            self.notifications.append(registration)

    async def offload(self, registration: TenantRegistration) -> None:
        async with self._lock:
            handle = next(item[0] for item in self.live.values() if item[1] == registration)
            self._callback_owner = asyncio.current_task()
            try:
                await handle.offload()
            finally:
                self._callback_owner = None

    async def evict(self, registration: TenantRegistration) -> None:
        async with self._lock:
            handle = next(item[0] for item in self.live.values() if item[1] == registration)
            self._callback_owner = asyncio.current_task()
            try:
                await handle.evict()
            finally:
                self._callback_owner = None

    def allocate(self, registration: TenantRegistration, device: str) -> None:
        if registration.assigned_device != device:
            raise RuntimeError("allocation is not on the assigned device")


def test_fake_tenant_registry_enforces_all_amendment_contracts() -> None:
    async def scenario() -> None:
        registry = _StrictFakeRegistry(budget=10)
        owned = _PackTenantRegistry("pack", registry)
        first = _Handle("pack", "model", 6)
        registration = await owned.register(first)
        await owned.notify_resident(registration)
        assert registry.notifications == [registration]
        with pytest.raises(CompositionError, match="pack.*unowned"):
            await owned.notify_resident(TenantRegistration("foreign", "cuda:1"))
        with pytest.raises(RuntimeError, match="duplicate"):
            await registry.register(first)
        registry.allocate(registration, "cuda:1")
        with pytest.raises(RuntimeError, match="assigned device"):
            registry.allocate(registration, "cuda:0")
        first.registry = registry
        with pytest.raises(RuntimeError, match="non-reentrant"):
            await registry.offload(registration)
        first.registry = None
        first.size_estimate_bytes = 100
        assert registry.live[("pack", "model")][2] == 6
        await owned.unregister(registration)
        with pytest.raises(RuntimeError, match="budget refused"):
            await owned.register(first)
        first.size_estimate_bytes = 4
        registration = await owned.register(first)
        await owned.notify_resident(registration)
        await registry.evict(registration)  # severe pressure may evict before offload
        await registry.evict(registration)  # callbacks must tolerate retries
        assert first.evictions == 2

        started = asyncio.Event()
        release = asyncio.Event()
        first.started = started
        first.release = release
        callback = asyncio.create_task(registry.offload(registration))
        await started.wait()
        unregister = asyncio.create_task(owned.unregister(registration))
        await asyncio.sleep(0)
        assert not unregister.done()  # unregister serializes behind the callback
        release.set()
        await callback
        await unregister
        assert registry.live == {}
        await owned.close()
        with pytest.raises(CompositionError, match="removed pack"):
            await owned.register(_Handle("pack", "late", 1))

        deadlock_registry = _StrictFakeRegistry()
        deadlock_owned = _PackTenantRegistry("pack", deadlock_registry)
        deadlock_started = asyncio.Event()
        deadlock_release = asyncio.Event()
        deadlock_handle = _Handle(
            "pack",
            "callback",
            1,
            registry=deadlock_owned,
            started=deadlock_started,
            release=deadlock_release,
        )
        deadlock_registration = await deadlock_owned.register(deadlock_handle)
        deadlock_callback = asyncio.create_task(deadlock_registry.offload(deadlock_registration))
        await deadlock_started.wait()
        closing = asyncio.create_task(deadlock_owned.close())
        await asyncio.sleep(0)
        deadlock_release.set()
        with pytest.raises(CompositionError, match="removed pack"):
            await asyncio.wait_for(deadlock_callback, 1)
        await asyncio.wait_for(closing, 1)
        assert deadlock_registry.live == {}

    asyncio.run(scenario())
