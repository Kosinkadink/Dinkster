"""H3.1 in-process hosting mechanics and tenant-contract proof."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_graph import Graph, GraphNode
from dinkster_memory import ModelTenantHandle, TenantRegistration
from dinkster_registry import InstallError, LockedPack, Lockfile
from dinkster_server import PackInfo, create_app
from dinkster_workers import ManifestError
from dinkster_workers.doctor import DoctorReport, Finding

import dinkster.installer as installer_module
from dinkster.compose import (
    PackSpec,
    ServingComposer,
    compose_serving,
)
from dinkster.installer import (
    Installer,
    _decode_hosting_record,
    lock_local_pack,
)
from tests.async_scenarios import run_scenario

PINS = {"torch": "2.13.0", "comfy-aimdo": "0.4.13"}


def _lockfile() -> Lockfile:
    return Lockfile(
        packs=(
            LockedPack("alpha", "1.0.0", "blake3:" + "1" * 64, "pub", ("alpha",), "registry:x"),
            LockedPack("beta", "1.0.0", "blake3:" + "2" * 64, "pub", ("beta",), "registry:x"),
        )
    )


def _manifest(directory: Path, *, extra: str = "") -> Path:
    directory.mkdir()
    path = directory / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "isopack"\npublisher = "dinkster"\n'
        'type-namespaces = ["iso"]\nnamespaces = ["iso"]\n'
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
        "STRING=TypeExpr.concrete('STRING')\n"
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
        '[pack]\nname = "h31-placement"\npublisher = "dinkster"\n'
        'namespaces = ["h31"]\n[pack.entry]\n'
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
                '[pack]\nname = "h31-second"\npublisher = "dinkster"\n'
                'namespaces = ["h32"]\n'
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
                '[pack]\nname = "h31-conflicting"\npublisher = "dinkster"\n'
                'namespaces = ["h33"]\n'
                '[pack.entry]\nnodes = "h31_second_nodes:NODES"\n'
            )
            with pytest.raises(ManifestError, match="already loaded from outside the pack root"):
                await composer.add_pack(PackSpec(conflicting, in_process=True, runtime_pins=PINS))
        finally:
            await composer.close()
            await isolated.close()

    run_scenario(scenario())


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
    with pytest.raises(InstallError, match="requires exact torch and comfy-aimdo"):
        installer.apply(
            target,
            hosting_groups=(),
            hosting_in_process=("alpha",),
            hosting_runtime_pins={},
        )

    aimdo = _install_pack(tmp_path / "aimdo", "alpha", requires='"comfy-aimdo==0.4.13"')
    aimdo_target = Lockfile.of([lock_local_pack(aimdo, installer.artifacts_dir)[0]])
    with pytest.raises(InstallError, match="must not declare a comfy-aimdo dependency"):
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
