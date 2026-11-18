"""Managed scratch storage is distinct from retained data and transport endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode
from dinkster_supervisor.environment_scratch import prepare_scratch_storage
from dinkster_supervisor.environment_state import EnvironmentError
from dinkster_workers import SandboxPolicy, detect_bubblewrap, detect_seatbelt
from dinkster_workers.windows_sandbox import detect_windows_sandbox
from test_compose_dispatch import _write_pack
from test_environment_runtime import free_port

from dinkster.compose import PackSpec, ServingComposer


def _require_sandbox() -> None:
    detect = (
        detect_bubblewrap
        if sys.platform == "linux"
        else detect_seatbelt
        if sys.platform == "darwin"
        else detect_windows_sandbox
    )
    capability = detect()
    if not capability.available:
        pytest.skip(capability.detail)


def _temporary_probe_pack(root: Path, name: str, sibling: Path) -> Path:
    manifest = _write_pack(root, name, name + ".echo")
    module = manifest.parent / (name + "_nodes.py")
    module.write_text(
        module.read_text().replace(
            f'return cls.outputs(value={name!r} + ":" + value)',
            "import json, os, sys, tempfile\n"
            "        from pathlib import Path\n"
            "        if sys.platform == 'win32':\n"
            "            import ctypes\n"
            "            selected = Path(os.environ['TMPDIR'])\n"
            "            assert Path(os.environ['TEMP']) == selected\n"
            "            assert Path(os.environ['TMP']) == selected\n"
            "            buffer = ctypes.create_unicode_buffer(32768)\n"
            "            length = ctypes.windll.kernel32.GetTempPathW(len(buffer), buffer)\n"
            "            assert 0 < length < len(buffer)\n"
            "            assert Path(buffer.value) == selected\n"
            "        with tempfile.NamedTemporaryFile(delete=False) as output:\n"
            "            output.write(value.encode())\n"
            "            path = Path(output.name)\n"
            f"        visible = Path({str(sibling)!r}).exists()\n"
            "        report = [str(path), path.read_text(), visible]\n"
            "        return cls.outputs(value=json.dumps(report))",
        )
    )
    return manifest


@pytest.mark.parametrize("sandbox", [False, True])
def test_managed_start_uses_saved_worker_storage_and_refuses_unavailable_storage(
    tmp_path: Path,
    sandbox: bool,
) -> None:
    import venv

    import aiohttp
    from dinkster_graph import graph_to_wire
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_supervisor.environment_runtime import start_engine, stop_engine
    from dinkster_workers.doctor import prepare_catalog
    from test_environment_runtime import make_generation
    from test_serve import _job_finish, _peek_value

    from dinkster.generation_packs import stage_generation_packs

    if sandbox:
        _require_sandbox()
    project, selected = tmp_path / "project", tmp_path / "selected"
    selected.mkdir()
    sibling = selected / "secret.txt"
    sibling.write_text("host sibling")
    manifest = _temporary_probe_pack(tmp_path / "source", "scratch", sibling)
    manifest.write_text(
        manifest.read_text().replace("[pack.entry]", 'version="1.0.0"\nengine=">=0"\n[pack.entry]')
    )
    archive = tmp_path / "pack.zip"
    digest = build_artifact(manifest.parent, archive)
    generation = project / "builds/1"
    generation.mkdir(parents=True)
    stage_generation_packs(
        generation,
        Lockfile.of([LockedPack("scratch", "1.0.0", digest, "dinkster", ("scratch",))]),
        {"scratch": archive},
    )
    make_generation(project, fake_serve=False)
    venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=os.name != "nt").create(
        generation / "packs/scratch"
    )
    source = generation / "pack-sources/scratch"
    assert prepare_catalog(
        source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
    ).ok
    port = free_port()
    (project / "environment.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": port,
                "scratchStorage": {"directory": str(selected)},
            }
        )
    )
    assigned = prepare_scratch_storage(selected, project, project / "dinkster-library")
    workers = assigned / "workers"

    async def probe() -> None:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession(trust_env=False) as client:
            async with client.post(
                base + "/api/jobs",
                json={
                    "clientId": "e2e",
                    "jobId": "scratch-probe",
                    "graph": graph_to_wire(
                        Graph(
                            nodes={"save": GraphNode("scratch.echo", {"value": "selected bytes"})}
                        )
                    ),
                    "targets": ["save"],
                },
            ) as response:
                assert response.status == 202, await response.text()
            assert (await _job_finish(client, base, "scratch-probe"))["state"] == "completed"
            report = await _peek_value(client, base, "scratch-probe", "value")
            assert isinstance(report, str)
            path, value, visible = json.loads(report)
            assert value == "selected bytes" and visible is (not sandbox)
            assert Path(path).is_relative_to(
                Path("/tmp") if sandbox and sys.platform == "linux" else workers
            )
            assert any(
                item.read_text() == value for item in workers.rglob("tmp*") if item.is_file()
            )

    try:
        start_engine(project, 1, "main", port, sandbox_packs=sandbox)
        asyncio.run(probe())
    finally:
        stop_engine(project, "main")
    assert list(workers.iterdir()) == [] and sibling.read_text() == "host sibling"
    offline = tmp_path / "offline"
    selected.rename(offline)
    try:
        with pytest.raises(EnvironmentError, match="unavailable"):
            start_engine(project, 1, "main", port, sandbox_packs=sandbox)
    finally:
        offline.rename(selected)
        stop_engine(project, "main")


def test_managed_scratch_configuration_preserves_retained_data(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    settings = project / "environment.json"
    settings.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": free_port(),
            }
        )
    )
    retained = project / "dinkster-library" / "outputs" / "kept.txt"
    retained.parent.mkdir(parents=True)
    retained.write_text("retained output")
    selected = tmp_path / "selected"
    selected.mkdir()
    unrelated = selected / "unrelated.txt"
    unrelated.write_text("not owned")
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(project),
        "scratch",
    ]
    before = settings.read_bytes()
    plan = subprocess.run(
        [*command, "--directory", str(selected), "--plan", "--json"], capture_output=True, text=True
    )
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert settings.read_bytes() == before and list(selected.iterdir()) == [unrelated]
    applied = subprocess.run(
        [*command, "--directory", str(selected), "--yes", "--json"], capture_output=True, text=True
    )
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assigned = Path(json.loads(applied.stdout)["scratchRoot"])
    pending = assigned / "packs" / "pending-handoff.txt"
    pending.write_text("pending output")
    assert json.loads(settings.read_text())["scratchStorage"] == {"directory": str(selected)}
    unavailable = subprocess.run(
        [*command, "--directory", str(tmp_path / "missing"), "--yes", "--json"],
        capture_output=True,
        text=True,
    )
    assert unavailable.returncode != 0 and "unavailable" in unavailable.stdout
    assert json.loads(settings.read_text())["scratchStorage"] == {"directory": str(selected)}
    cleared = subprocess.run([*command, "--yes", "--json"], capture_output=True, text=True)
    assert cleared.returncode == 0, cleared.stdout + cleared.stderr
    assert json.loads(settings.read_text())["scratchStorage"] == {}
    assert pending.read_text() == "pending output"
    assert retained.read_text() == "retained output" and unrelated.read_text() == "not owned"


def test_scratch_refuses_redirected_or_unowned_roots(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    project, library = tmp_path / "project", tmp_path / "library"
    assigned = prepare_scratch_storage(selected, project, library)
    (assigned / "owner.json").unlink()
    with pytest.raises(EnvironmentError, match="not owned"):
        prepare_scratch_storage(selected, project, library)
    if os.name != "nt":
        linked = tmp_path / "linked"
        linked.symlink_to(selected, target_is_directory=True)
        with pytest.raises(EnvironmentError, match="linked"):
            prepare_scratch_storage(linked, project, library)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("sandbox", [False, True])
def test_worker_temporary_bytes_use_private_selected_storage_and_cleanup(
    tmp_path: Path,
    grouped: bool,
    sandbox: bool,
) -> None:
    if sandbox:
        _require_sandbox()
    selected = tmp_path / ("long-selected-directory-" * 8)
    selected.mkdir()
    sibling = selected / "host-secret.txt"
    sibling.write_text("host secret")
    workers = selected / "workers"
    workers.mkdir()
    manifests = [
        _temporary_probe_pack(tmp_path / name, name, sibling)
        for name in (("scratchone", "scratchtwo") if grouped else ("scratchone",))
    ]

    async def scenario() -> None:
        composer = ServingComposer(
            worker_temporary_root=workers,
            sandbox_policy=SandboxPolicy() if sandbox else None,
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest,
                        worker_group="shared" if grouped else None,
                        group_manifests=tuple(manifests) if grouped else (),
                    )
                )
            engine = composer.composition.make_engine(lambda event: None)
            for name in ("scratchone", "scratchtwo") if grouped else ("scratchone",):
                result = await engine.run(
                    Graph(nodes={"probe": GraphNode(name + ".echo", {"value": name})}), ["probe"]
                )
                report = result.outputs["probe"]["value"].resolve()
                assert isinstance(report, str)
                path, value, visible = json.loads(report)
                assert value == name and visible is (not sandbox)
                expected = Path("/tmp") if sandbox and sys.platform == "linux" else workers
                assert Path(path).is_relative_to(expected)
                assert any(
                    item.name.startswith(("worker-", "group-")) for item in workers.iterdir()
                )
                assert any(
                    item.read_text() == name for item in workers.rglob("tmp*") if item.is_file()
                )
        finally:
            await composer.close()
        assert list(workers.iterdir()) == []
        assert sibling.read_text() == "host secret"

    asyncio.run(scenario())
