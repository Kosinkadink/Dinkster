"""Activation keeps real supervisors recoverable on both sides of current replacement."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

import pytest
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_supervisor.environment_runtime import engine_status, start_engine, stop_engine
from dinkster_supervisor.environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    content_digest,
    interpreter,
    read_current,
    write_document,
)
from dinkster_supervisor.environment_stores import (
    register_project,
    registry_path,
    setup_registration,
    volume,
)
from dinkster_supervisor.environment_transactions import (
    activate,
    begin_activation,
    begin_setup,
    recover_activation,
)
from test_environment_artifacts import _mirror
from test_environment_cli import _run
from test_environment_generations import _standalone_tree
from test_environment_runtime import free_port, make_generation


def _target(root: Path, number: int) -> Path:
    make_generation(root, number)
    target = root / "generations" / str(number)
    shutil.rmtree(target / "python")
    _standalone_tree(target / "python")
    generation = Generation.read(target / "generation.json")
    hashes = tuple(
        sorted(
            (path.relative_to(target).as_posix(), content_digest(path))
            for path in target.rglob("*")
            if path.is_file() and path.name != "generation.json"
        )
    )
    replace(generation, constraints=content_digest(target / "constraints.txt"), files=hashes).write(
        target / "generation.json"
    )
    return target


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("later_project", [False, True])
def test_interrupted_setup_metadata_recovery_preserves_cache_settings_and_other_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, later_project: bool
) -> None:
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    root, store, other = (tmp_path / name for name in ("project", "store", "other"))
    root.mkdir()
    store.mkdir()
    registry = registry_path()
    registry.parent.mkdir(parents=True)
    if existing:
        (root / "project.json").write_bytes(b'{"custom": "preserve exact bytes"}\r\n')
        (root / "store").write_bytes((str(store) + "\r\n").encode())
        write_document(
            store / "projects.json", {"format": "dinkster.store/1", "projects": [str(other)]}
        )
        write_document(
            registry,
            {
                "format": "dinkster.stores/1",
                "volumes": {volume(root): str(store), "999999": str(tmp_path / "unrelated")},
            },
        )
    before = {
        name: (root / name).read_bytes() if (root / name).exists() else None
        for name in ("project.json", "store")
    }
    path = begin_setup(root, 1, free_port(), registration=setup_registration(root, store))
    register_project(root, store)
    (root / "project.json").write_text('{"custom": "setup write"}')
    (root / "store").write_text(str(store) + "\n")
    (store / "cached-wheel").write_bytes(b"keep verified cache")
    script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment, environment_stores
original = environment_stores.durable_unlink
original_write = environment_stores.write_document
def interrupt(path):
    original(path)
    os._exit(73)
def interrupt_write(path, value):
    original_write(path, value)
    os._exit(73)
environment_stores.durable_unlink = interrupt
environment_stores.write_document = interrupt_write
environment._recover_pending(Path(sys.argv[1]))
"""
    interrupted = subprocess.run([sys.executable, "-c", script, str(root)], timeout=30)
    assert interrupted.returncode == 73
    assert Journal.read(path).incomplete
    if later_project:
        register_project(other, store)
    from dinkster_supervisor.environment import _recover_pending

    _recover_pending(root)
    _recover_pending(root)
    assert Journal.read(path).stage == "rolled-back"
    assert (store / "cached-wheel").read_bytes() == b"keep verified cache"
    assert not (root / "current").exists()
    for name, contents in before.items():
        assert ((root / name).read_bytes() if (root / name).exists() else None) == contents
    if existing or later_project:
        assert json.loads((store / "projects.json").read_bytes())["projects"] == [str(other)]
        expected = {volume(root): str(store)}
        if existing:
            expected["999999"] = str(tmp_path / "unrelated")
        assert json.loads(registry.read_bytes())["volumes"] == expected
    else:
        assert not (store / "projects.json").exists()
        assert not registry.exists()


@pytest.mark.parametrize("interruption", [None, "before-switch", "after-switch"])
def test_first_setup_publishes_only_a_healthy_engine_and_recovers_without_previous_generation(
    tmp_path: Path, interruption: str | None
) -> None:
    port = free_port()
    path = begin_setup(tmp_path, 1, port, ("GPU-aB12",), name="compute")
    assert read_current(tmp_path) is None
    assert Journal.read(path).previous is None
    _target(tmp_path, 1)
    (tmp_path / "current").unlink()
    script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment_transactions as transactions
root, journal = Path(sys.argv[1]), Path(sys.argv[2])
def interrupted():
    os._exit(71)
if sys.argv[3] == 'after-switch':
    original = transactions.set_current
    def switch_then_interrupt(root, number):
        original(root, number)
        interrupted()
    transactions.set_current = switch_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
else:
    transactions.activate(root, journal, revalidate=interrupted)
"""
    try:
        if interruption is None:
            activate(tmp_path, path, revalidate=lambda: None)
            assert read_current(tmp_path) == 1
            assert engine_status(tmp_path) == (
                {
                    "name": "compute",
                    "generation": 1,
                    "state": "ready",
                    "url": f"http://127.0.0.1:{port}",
                },
            )
            with urlopen(f"http://127.0.0.1:{port}/api/cuda") as response:  # noqa: S310
                assert json.loads(response.read()) == {"cuda": "GPU-aB12"}
            assert Journal.read(path).stage == "complete"
            with pytest.raises(EnvironmentError, match="already has an active"):
                begin_setup(tmp_path, 2, free_port())
        else:
            exited = subprocess.run(
                [sys.executable, "-c", script, str(tmp_path), str(path), interruption],
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert exited.returncode == 71, exited.stderr
            assert Journal.read(path).incomplete
            recover_activation(tmp_path, path)
            assert read_current(tmp_path) is None
            assert engine_status(tmp_path) == (
                {
                    "name": "compute",
                    "generation": 1,
                    "state": "stopped",
                    "url": f"http://127.0.0.1:{port}",
                },
            )
            assert not (tmp_path / "generations/1").exists()
            assert Journal.read(path).stage == "rolled-back"
            recover_activation(tmp_path, path)
            next_path = begin_setup(tmp_path, 1, port)
            assert Journal.read(next_path).incomplete
            recover_activation(tmp_path, next_path)
    finally:
        stop_engine(tmp_path, "compute", generation=1)


@pytest.mark.parametrize("migration", [False, True])
def test_update_and_rollback_reactivate_retained_tree_without_rebuild(
    tmp_path: Path, migration: bool
) -> None:
    retained = _target(tmp_path, 1)
    before = (retained / "generation.json").read_bytes()
    identity = interpreter(retained).stat()
    (tmp_path / "project.json").write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "main",
                "variant": "linux-cpu",
                "port": free_port(),
                "mirrors": [str(tmp_path / "mirror")],
            }
        )
    )
    start_engine(tmp_path, 1, "main", free_port(), ("GPU-aB12",))
    parked_port = free_port()
    start_engine(tmp_path, 1, "parked", parked_port)
    parked_path = tmp_path / "runtime/1/parked.json"
    stale_parked = parked_path.read_bytes()
    stop_engine(tmp_path, "parked", generation=1)
    parked_path.write_bytes(stale_parked)
    previous_settings = (tmp_path / "project.json").read_text()
    selected = json.loads(previous_settings)
    selected.update(channel="stable", variant="win-cpu", custom={"keep": "selection"})
    next_settings = json.dumps(selected) + "\n"
    path = begin_activation(
        tmp_path,
        2,
        "update",
        project_settings={"previous": previous_settings, "target": next_settings}
        if migration
        else None,
    )
    _target(tmp_path, 2)

    def revalidate() -> None:
        assert read_current(tmp_path) == 1
        assert (tmp_path / "project.json").read_text() == previous_settings
        binding = next(row for row in Journal.read(path).bindings if row["name"] == "main")
        with urlopen(f"http://127.0.0.1:{binding['targetPort']}/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(tmp_path / "generations/2")}

    try:
        activate(tmp_path, path, revalidate=revalidate)
        assert read_current(tmp_path) == 2
        assert (tmp_path / "project.json").read_text() == (
            next_settings if migration else previous_settings
        )
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert engine_status(tmp_path)[0]["generation"] == 2
        assert engine_status(tmp_path)[1]["url"] == f"http://127.0.0.1:{parked_port}"
        assert Journal.read(path).stage == "complete"
        assert (tmp_path / "generations/1").is_dir()
        code, restarted = _run(tmp_path, "start", "--engine", "parked", "--yes")
        assert code == 0 and restarted["url"] == f"http://127.0.0.1:{parked_port}"
        assert _run(tmp_path, "stop", "--engine", "parked", "--yes")[0] == 0
        journals = sorted((tmp_path / "journal").glob("*.json"))
        if migration:
            (tmp_path / "project.json").write_text(json.dumps({**selected, "custom": "later edit"}))
            edited = (tmp_path / "project.json").read_bytes()
            code, refused = _run(tmp_path, "rollback", "--yes")
            assert code == 1 and "settings changed since selection" in refused["error"]
            assert (tmp_path / "project.json").read_bytes() == edited
            assert read_current(tmp_path) == 2
            assert sorted((tmp_path / "journal").glob("*.json")) == journals
            (tmp_path / "project.json").write_bytes(next_settings.encode("utf-8"))
        code, planned = _run(tmp_path, "rollback", "--plan")
        assert code == 0 and planned["status"] == "planned"
        assert read_current(tmp_path) == 2
        assert sorted((tmp_path / "journal").glob("*.json")) == journals
        code, applied = _run(tmp_path, "rollback", "--yes")
        assert code == 0, applied
        assert applied["status"] == "applied"
        assert read_current(tmp_path) == 1
        assert (tmp_path / "project.json").read_text() == previous_settings
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert engine_status(tmp_path)[1]["state"] == "stopped"
        code, restarted = _run(tmp_path, "start", "--engine", "parked", "--yes")
        assert code == 0 and restarted["url"] == f"http://127.0.0.1:{parked_port}"
        assert _run(tmp_path, "stop", "--engine", "parked", "--yes")[0] == 0
        with urlopen(str(applied["engines"][0]["url"]) + "/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(tmp_path / "generations/1")}
        assert (retained / "generation.json").read_bytes() == before
        after = interpreter(retained).stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
            identity.st_ino,
            identity.st_size,
            identity.st_mtime_ns,
        )
        assert (tmp_path / "generations/2").is_dir()
        (tmp_path / "generations/2/constraints.txt").write_text("changed\n")
        code, refused = _run(tmp_path, "rollback", "--yes")
        assert code == 1 and "drift" in refused["error"]
        assert read_current(tmp_path) == 1
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert (tmp_path / "generations/2").is_dir()
    finally:
        stop_engine(tmp_path, "main", generation=1)
        stop_engine(tmp_path, "main", generation=2)
        stop_engine(tmp_path, "parked", generation=1)
        stop_engine(tmp_path, "parked", generation=2)


@pytest.mark.parametrize(
    "interruption", ["none", "unavailable", "asset-removed", "workflow-edited"]
)
@pytest.mark.parametrize("migration", [False, True])
def test_confirmed_workflow_update_checks_inputs_after_candidate_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interruption: str,
    migration: bool,
) -> None:
    from dinkster_supervisor import environment, environment_transactions

    from dinkster.generation_packs import stage_generation_packs

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(str(Path(p).resolve()) for p in sys.path))
    root, store, mirror = (tmp_path / name for name in ("project", "store", "mirror"))
    _mirror(mirror, variant="win-cpu" if migration else "linux-cpu")
    if migration:
        shutil.copyfile(mirror / "channels/main.json", mirror / "channels/stable.json")
    old = _target(root, 1)
    (old / "dinkster.lock").write_text(Lockfile().record_json())
    (root / "store").write_text(str(store) + "\n")
    payload = b"verified model from selected pack"
    identity = digest_bytes(payload)
    asset = AssetRef(
        digest_bytes(b"unavailable") if interruption == "unavailable" else identity,
        "model.bin",
        len(payload),
    )
    vault = AssetVault(root / "library/vault")
    document = {
        "format": "dinkster-workflow",
        "formatVersion": 1,
        "graphs": {"main": {"nodes": {}, "valueSources": {"model": {"value": asset.to_wire()}}}},
    }
    data = json.dumps(document).encode()
    workflow_digest = digest_bytes(data)
    with vault.writer(workflow_digest) as writer:
        writer.write(data)
        writer.commit()
    library = LibraryStore(root / "library/library.sqlite")
    record = library.create(
        "local", "Workflow", workflow_digest, "application/x-dinkster-workflow+json"
    )
    library.close()
    port = free_port()
    settings = {
        "format": "dinkster.project/1",
        "channel": "main",
        "variant": "linux-cpu",
        "mirrors": [str(mirror)],
        "port": port,
        "workflows": [{"scope": "local", "id": record.id}],
        "additions": {"packs": [{"pack": "example", "version": "1.0.0"}]},
    }
    (root / "project.json").write_text(json.dumps(settings))
    original_settings = (root / "project.json").read_bytes()
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text(
        '[pack]\nname = "example"\n[pack.entry]\nnodes = "poison:NODES"\n'
        '\n[[pack.assets]]\nid = "model"\nname = "Model"\n'
        f'digest = "{identity}"\nfile = "model.bin"\n'
    )
    (pack / "model.bin").write_bytes(payload)
    (pack / "poison.py").write_text('raise RuntimeError("must not import pack nodes")\n')
    archives = tmp_path / "artifacts"
    archives.mkdir()
    archive = archives / "building.zip"
    artifact_digest = build_artifact(pack, archive)
    archive = archive.rename(archives / f"{artifact_digest.split(':')[1]}.zip")
    lock = Lockfile.of([LockedPack("example", "1.0.0", artifact_digest, "author", ("example",))])
    catalog = tmp_path / "official-packs.lock"
    catalog.write_text(lock.record_json())
    core = Path(__file__).resolve().parents[1] / "src/dinkster"

    def fixture_build(**kwargs: object) -> None:
        # Admission, acquisition and activation use the real paths over fixture generations.
        target = _target(root, int(str(kwargs["number"])))
        generation = Generation.read(target / "generation.json")
        (target / "generation.json").unlink()
        stage_generation_packs(target, lock, {"example": archive})
        files = tuple(
            sorted(
                (p.relative_to(target).as_posix(), content_digest(p))
                for p in target.rglob("*")
                if p.is_file()
            )
        )
        replace(
            generation, files=files, in_process_packs=("example",), variant=str(kwargs["variant"])
        ).write(target / "generation.json")
        (root / "dinkster/__init__.py").write_text(f"__path__.append({str(core)!r})\n")

    monkeypatch.setattr(environment, "build_generation", fixture_build)
    (root / "dinkster/__init__.py").write_text(f"__path__.append({str(core)!r})\n")
    prior = start_engine(root, 1, "main", port)
    original_start = environment_transactions.start_engine

    def candidate_start(
        root: Path,
        generation: int,
        name: str,
        port: int,
        gpus: tuple[str, ...] = (),
    ) -> dict[str, object]:
        result = original_start(root, generation, name, port, gpus)
        if generation != 2:
            return result
        if interruption == "asset-removed":
            replace(asset, resolver=vault).local_path().unlink()
        elif interruption == "workflow-edited":
            index = LibraryStore(root / "library/library.sqlite")
            try:
                index.update("local", record.id, record.revision, name="Concurrent edit")
            finally:
                index.close()
        return result

    monkeypatch.setattr(environment_transactions, "start_engine", candidate_start)
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "dinkster-env",
                "--root",
                str(root),
                "update",
                "--yes",
                "--json",
                "--catalog-lock",
                str(catalog),
                "--artifacts",
                str(archives),
                *(["--channel", "stable", "--variant", "win-cpu"] if migration else []),
            ],
        )
        with pytest.raises(SystemExit) as exited:
            environment.main()
        code = exited.value.code
        result = json.loads(capsys.readouterr().out)
        journals = list((root / "journal").glob("*.json"))
        assert len(journals) == 1
        if interruption == "none":
            assert code == 0, result
            assert result["status"] == "applied" and read_current(root) == 2
            assert (
                AssetRef(identity, "model.bin", len(payload), resolver=vault).read_bytes()
                == payload
            )
            assert Journal.read(journals[0]).stage == "complete"
            if migration:
                assert json.loads((root / "project.json").read_bytes()) == {
                    **settings,
                    "channel": "stable",
                    "variant": "win-cpu",
                }
                index = LibraryStore(root / "library/library.sqlite")
                try:
                    assert index.get("local", record.id) == record
                finally:
                    index.close()
        else:
            assert code == 1, result
            expected = (
                "no usable sources"
                if interruption == "unavailable"
                else "requirements failed revalidation"
                if interruption == "asset-removed"
                else "workflow inputs changed"
            )
            assert expected in result["error"]
            assert read_current(root) == 1
            assert engine_status(root) == (prior,)
            assert not (root / "generations/2").exists()
            assert Journal.read(journals[0]).stage == "rolled-back"
            with urlopen(f"{prior['url']}/api/health") as response:  # noqa: S310
                assert json.loads(response.read()) == {"ok": True}
        if interruption != "none" or not migration:
            assert (root / "project.json").read_bytes() == original_settings
        assert (old / "dinkster.lock").read_text() == Lockfile().record_json()
    finally:
        stop_engine(root, "main", generation=1)
        stop_engine(root, "main", generation=2)


@pytest.mark.parametrize(
    "interruption",
    [
        "before-spawn",
        "before-switch",
        "after-settings",
        "after-switch",
        "after-shutdown",
        "during-recovery",
    ],
)
def test_interrupted_activation_restores_previous_serving_generation(
    tmp_path: Path,
    interruption: str,
) -> None:
    make_generation(tmp_path)
    prior = start_engine(tmp_path, 1, "main", free_port())
    previous_settings = (
        '{"format":"dinkster.project/1","channel":"main","variant":"linux-cpu","custom":42}\r\n'
    )
    target_settings = (
        '{"format":"dinkster.project/1","channel":"stable","variant":"win-cpu","custom":42}\n'
    )
    (tmp_path / "project.json").write_bytes(previous_settings.encode())
    path = begin_activation(
        tmp_path,
        2,
        "update",
        project_settings={"previous": previous_settings, "target": target_settings},
    )
    _target(tmp_path, 2)

    script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment_transactions as transactions
from dinkster_supervisor import environment_runtime as runtime
root, journal = Path(sys.argv[1]), Path(sys.argv[2])
def interrupted():
    os._exit(71)
if sys.argv[3] == 'after-settings':
    original = transactions.atomic_write
    def settings_then_interrupt(path, text):
        original(path, text)
        interrupted()
    transactions.atomic_write = settings_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
elif sys.argv[3] == 'after-switch':
    original = transactions.set_current
    def switch_then_interrupt(root, number):
        original(root, number)
        interrupted()
    transactions.set_current = switch_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
elif sys.argv[3] in {'before-switch', 'during-recovery'}:
    transactions.activate(root, journal, revalidate=interrupted)
elif sys.argv[3] == 'before-spawn':
    runtime.subprocess.Popen = lambda *a, **kw: interrupted()
    transactions.activate(root, journal, revalidate=lambda: None)
else:
    original = runtime._write_record
    def stopped_then_interrupt(root, name, record):
        if record['state'] == 'stopped' and record['generation'] == 1:
            interrupted()
        original(root, name, record)
    runtime._write_record = stopped_then_interrupt
    transactions.activate(root, journal, revalidate=lambda: None)
"""
    try:
        exited = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), str(path), interruption],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert exited.returncode == 71, exited.stderr
        assert Journal.read(path).incomplete
        if interruption in {"after-settings", "after-switch", "after-shutdown"}:
            assert (tmp_path / "project.json").read_bytes() == target_settings.encode()
        if interruption == "during-recovery":
            recovery_script = """
import os
import sys
from pathlib import Path
from dinkster_supervisor import environment_runtime as runtime
from dinkster_supervisor.environment_transactions import recover_activation
original = runtime._write_record
def stopped_then_interrupt(root, name, record):
    if record['state'] == 'stopped' and record['generation'] == 2:
        os._exit(72)
    original(root, name, record)
runtime._write_record = stopped_then_interrupt
recover_activation(Path(sys.argv[1]), Path(sys.argv[2]))
"""
            exited = subprocess.run(
                [sys.executable, "-c", recovery_script, str(tmp_path), str(path)],
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert exited.returncode == 72, exited.stderr
            assert Journal.read(path).incomplete
        if interruption in {"after-settings", "after-shutdown"}:
            external = target_settings.replace('"custom":42', '"custom":99').encode()
            (tmp_path / "project.json").write_bytes(external)
            current_before_refusal = read_current(tmp_path)
            serving_before_refusal = engine_status(tmp_path)
            with pytest.raises(EnvironmentError, match="settings changed during interrupted"):
                recover_activation(tmp_path, path)
            assert (tmp_path / "project.json").read_bytes() == external
            assert read_current(tmp_path) == current_before_refusal
            assert engine_status(tmp_path) == serving_before_refusal
            assert serving_before_refusal[0]["state"] == "ready"
            with urlopen(str(serving_before_refusal[0]["url"]) + "/api/health") as response:  # noqa: S310
                assert json.loads(response.read()) == {"ok": True}
            assert Journal.read(path).incomplete
            (tmp_path / "project.json").write_bytes(target_settings.encode())
        recover_activation(tmp_path, path)
        assert read_current(tmp_path) == 1
        assert (tmp_path / "project.json").read_bytes() == previous_settings.encode()
        assert engine_status(tmp_path) == (prior,)
        assert not (tmp_path / "generations/2").exists()
        assert Journal.read(path).stage == "rolled-back"
        recover_activation(tmp_path, path)
        assert engine_status(tmp_path) == (prior,)
    finally:
        for number in (1, 2):
            with suppress(EnvironmentError):
                stop_engine(tmp_path, "main", generation=number)


def test_busy_refusal_and_failed_final_revalidation_preserve_current(tmp_path: Path) -> None:
    make_generation(tmp_path)
    prior = start_engine(tmp_path, 1, "main", free_port())
    try:
        (tmp_path / "queue.json").write_text('{"running":[{"jobId":"active"}],"queued":[]}')
        with pytest.raises(EnvironmentError, match="Finish or cancel"):
            begin_activation(tmp_path, 2, "update")
        assert not (tmp_path / "journal").exists()
        (tmp_path / "queue.json").write_text(json.dumps({"running": [], "queued": []}))
        settings = '{"format":"dinkster.project/1","channel":"main"}\n'
        (tmp_path / "project.json").write_text(settings)
        path = begin_activation(
            tmp_path,
            2,
            "update",
            project_settings={"previous": settings, "target": settings.replace("main", "stable")},
        )
        _target(tmp_path, 2)

        def changed() -> None:
            (tmp_path / "project.json").write_text(settings.replace("main", "external"))
            raise EnvironmentError("project changed after confirmation")

        with pytest.raises(EnvironmentError, match="project changed"):
            activate(tmp_path, path, revalidate=changed)
        assert (tmp_path / "project.json").read_text() == settings.replace("main", "external")
        assert read_current(tmp_path) == 1
        assert engine_status(tmp_path) == (prior,)
        assert not (tmp_path / "generations/2").exists()
        assert Journal.read(path).error == "project changed after confirmation"
    finally:
        stop_engine(tmp_path, "main", generation=1)
        stop_engine(tmp_path, "main", generation=2)
