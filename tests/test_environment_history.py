"""Installation history survives materialization loss and cannot replace earlier entries."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from dinkster_store import object_path, pack_archive_path
from dinkster_supervisor.environment_history import (
    activation_entries,
    capture_software,
    generation_history_coverage,
    history,
    import_generation,
    prepare_entry,
    publish_entry,
)
from dinkster_supervisor.environment_runtime import engine_status
from dinkster_supervisor.environment_state import (
    EnvironmentError,
    Generation,
    Journal,
    content_digest,
    interpreter,
    set_current,
    writer_lock,
)
from test_environment_cli import _run
from test_environment_runtime import free_port
from test_environment_state import _generation

from tests.fixtures.environments.build import build_base, canonical_json, executable_release, wheel


def test_installed_wheel_pack_add_and_update_without_index_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES
    from test_environment_generations import _workspace_tree
    from test_installer import _serve_artifacts, _static_release

    from dinkster.pack_archive import build_pack_archive
    from tests.fixtures.environments import build as publisher

    checkout = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    for project in ("dinkster", "dinkster-api", "dinkster-engine", "dinkster-supervisor"):
        subprocess.run(
            ["uv", "build", "--package", project, "--wheel", "--out-dir", str(wheels)],
            cwd=checkout,
            check=True,
        )
    host = tmp_path / "host"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(host)], check=True)
    python = host / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-sources",
            *map(str, wheels.glob("*.whl")),
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import dinkster, dinkster_api, dinkster_engine, dinkster_supervisor; "
            "from pathlib import Path; "
            f"assert all(Path(m.__file__).is_relative_to({str(host)!r}) for m in "
            "(dinkster, dinkster_api, dinkster_engine, dinkster_supervisor))",
        ],
        cwd=tmp_path,
        check=True,
    )
    original_wheel, encode = publisher.wheel, publisher.canonical_json
    monkeypatch.setattr(
        publisher, "wheel", lambda *args, **kwargs: original_wheel(*args, **kwargs, pack_tools=True)
    )
    monkeypatch.setattr(publisher, "standalone_tree", _workspace_tree)

    def encoded(document):
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        return encode(document)

    monkeypatch.setattr(publisher, "canonical_json", encoded)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = publisher.build(tmp_path / "fixture", "fixture-cpu", free_port())
    root = Path(fixture["root"])
    initial = json.loads(Path(fixture["initialChannel"]).read_bytes())
    empty = tmp_path / "empty.lock"
    empty.write_text(Lockfile.of([]).record_json())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    setup = fixture["commands"]["setup"]
    code, result = _run(
        root,
        *setup[setup.index("setup") : -1],
        "--catalog-lock",
        str(empty),
        "--artifacts",
        str(artifacts),
        timeout=120,
    )
    assert code == 0 and result["current"] == 1, result
    assert _run(root, "stop", "--yes")[0] == 0
    files: dict[str, bytes] = {}
    releases, entries = [], []
    for version, value in (("1.0.0", 31), ("1.1.0", 79)):
        pack = tmp_path / version
        pack.mkdir()
        (pack / "dinkster-pack.toml").write_text(
            HEALTHY_MANIFEST.replace('name = "healthy-nodes"', 'name = "healthy"')
            .replace("healthy_nodes:", "healthy:")
            .replace('requires = ["numpy>=1.26"]', 'requires = ["fixture-helper>=1"]')
            .replace("[pack]\n", f'[pack]\nversion = "{version}"\nengine = ">=1,<3"\n')
        )
        (pack / "healthy").mkdir()
        (pack / "healthy/__init__.py").write_text(HEALTHY_NODES + f"\nVALUE = {value}\n")
        (pack / "pyproject.toml").write_text(f'[project]\nname="healthy"\nversion="{version}"\n')
        archive = tmp_path / f"healthy-{version}-py3-none-any.whl"
        build_pack_archive(pack, archive)
        payload = archive.read_bytes()
        entry = LockedPack(
            "healthy",
            version,
            artifact_digest(payload),
            "dinkster",
            ("healthy",),
            "registry:official",
        )
        entries.append(entry)
        lock_path = tmp_path / f"{version}.pack-lock.json"
        dependency = wheel(tmp_path, "fixture_helper", version, value + 7)
        PackLock(
            "healthy",
            "fixture-cpu",
            ">=3.12",
            ("fixture-helper>=1",),
            (),
            (
                LockedDistribution(
                    "fixture-helper",
                    version,
                    (
                        LockedWheel(
                            dependency.name,
                            dependency.stat().st_size,
                            hashlib.sha256(dependency.read_bytes()).hexdigest(),
                            dependency.as_uri(),
                        ),
                    ),
                ),
            ),
        ).write(lock_path)
        lock = lock_path.read_bytes()
        lock_url = "locks/" + artifact_digest(lock).replace(":", "/") + ".json"
        release = {
            **_static_release(entry, payload),
            "engine": ">=1,<3",
            "installLocks": {
                "fixture-cpu": {
                    "baseId": initial["base"]["fixture-cpu"]["id"],
                    "artifactDigest": entry.artifact_digest,
                    "digest": artifact_digest(lock),
                    "size": len(lock),
                    "url": lock_url,
                }
            },
        }
        releases.append(release)
        files["/static/" + lock_url] = lock
        files["/static/" + release["wheels"][0]["url"]] = payload
    files["/static/packs/index.json"] = json.dumps({"schemaVersion": 1, "items": releases}).encode()
    requests = []
    server = _serve_artifacts(files, requests)
    trap = tmp_path / "trap"
    trap.mkdir()
    uv = shutil.which("uv")
    assert uv is not None
    log = trap / "uv.jsonl"
    wrapper = (
        "import json,subprocess,sys\n"
        "def main():\n"
        f"    with open({str(log)!r},'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "    a=sys.argv[1:]\n"
        "    assert a[:1] in (['pip'],['venv'],['cache']), 'unexpected uv operation'\n"
        "    if a[:1]==['pip']:\n"
        "        assert a[1] in ('install','sync'), 'package index resolver attempted'\n"
        "        assert '--offline' in a and '--no-index' in a, 'package index access attempted'\n"
        "        assert '--require-hashes' in a, 'unhashed install attempted'\n"
        "        if a[1]=='install': assert '--no-deps' in a, 'dependency resolution attempted'\n"
        f"    return subprocess.call([{uv!r},*a])\n"
    )
    guard = trap / "resolver_guard-1.0.0-py3-none-any.whl"
    metadata = "resolver_guard-1.0.0.dist-info"
    # A wheel entry point supplies uv.exe, which Windows CreateProcess finds unlike uv.cmd.
    with zipfile.ZipFile(guard, "w") as archive:
        archive.writestr("resolver_guard.py", wrapper)
        archive.writestr(
            f"{metadata}/METADATA", "Metadata-Version: 2.1\nName: resolver-guard\nVersion: 1.0.0\n"
        )
        archive.writestr(
            f"{metadata}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(
            f"{metadata}/entry_points.txt", "[console_scripts]\nuv = resolver_guard:main\n"
        )
        archive.writestr(f"{metadata}/RECORD", "")
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(guard)], check=True
    )
    monkeypatch.setenv("PATH", str(python.parent) + os.pathsep + os.environ["PATH"])
    control = subprocess.run(["uv", "pip", "compile", "absent"], capture_output=True, text=True)
    assert control.returncode != 0 and "package index resolver attempted" in control.stderr
    log.write_text("")
    (root / "registries.toml").write_text(
        f'[registries.official]\nendpoint="http://127.0.0.1:{server.server_port}"\ndefault=true\n'
    )
    # Linux tmpfs exercises a system TEMP on a different volume from the managed store.
    review_temp = tempfile.TemporaryDirectory(
        dir="/dev/shm" if sys.platform == "linux" and Path("/dev/shm").is_dir() else None
    )
    try:
        for args, number, entry, value in (
            (("install", "healthy@1.0.0"), 2, entries[0], 31),
            (("update", "healthy"), 3, entries[1], 79),
        ):
            command = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-m",
                    "dinkster.cli",
                    "pack",
                    "--root",
                    str(root),
                    *args,
                    "--yes",
                ],
                cwd=tmp_path,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=120,
                env={
                    **os.environ,
                    "TMPDIR": review_temp.name,
                    "TEMP": review_temp.name,
                    "TMP": review_temp.name,
                },
            )
            assert command.returncode == 0, command.stdout + command.stderr
            assert f"activated generation {number}" in command.stdout
            assert (root / "current").read_text() == f"{number}\n"
            generation = root / "generations" / str(number)
            software = Generation.read(generation / "generation.json")
            assert software.engine_release == initial["engine"]["release"]
            assert software.base == initial["base"]["fixture-cpu"]["id"]
            locked = Lockfile.from_record_json((generation / "dinkster.lock").read_text())
            assert locked.get("healthy") == entry
            pack_python = (
                generation
                / "packs/healthy"
                / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            )
            observed = subprocess.check_output(
                [
                    str(pack_python),
                    "-I",
                    "-c",
                    "import healthy, fixture_helper; print(healthy.VALUE, fixture_helper.VALUE)",
                ],
                text=True,
            )
            assert observed.strip() == f"{value} {value + 7}"
            assert _run(root, "stop", "--yes")[0] == 0
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(row[:2] == ["pip", "sync"] for row in commands), commands
        assert not any(row[:2] == ["pip", "compile"] for row in commands)
        assert requests and all(path.startswith("/static/") for path, _ in requests), requests
        request.node.add_report_section(
            "call",
            "Published pack resolver trap",
            f"launcher: {shutil.which('uv')}\n"
            f"compile negative control: {control.stderr}\n"
            f"guarded public add/update uv calls: {json.dumps(commands)}\n"
            "public add/update activated generations 2/3",
        )
    finally:
        _run(root, "stop", "--yes")
        server.shutdown()
        review_temp.cleanup()


@pytest.mark.parametrize("pack_route", ["supervisor", "command"])
def test_fresh_install_engine_update_pack_install_and_exact_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pack_route: str,
):
    from importlib.metadata import version

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.pack_lock import PackLock
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES
    from test_environment_generations import _workspace_tree
    from test_installer import _serve_artifacts, _static_indexes, _static_release, run_cli

    from dinkster.pack_archive import build_pack_archive
    from tests.fixtures.environments import build as publisher

    original_wheel, encode = publisher.wheel, publisher.canonical_json

    def published_wheel(*args, **kwargs):
        return original_wheel(*args, **kwargs, pack_tools=True)

    def canonical_json(document: dict) -> bytes:
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        return encode(document)

    monkeypatch.setattr(publisher, "wheel", published_wheel)
    monkeypatch.setattr(publisher, "standalone_tree", _workspace_tree)
    monkeypatch.setattr(publisher, "canonical_json", canonical_json)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = publisher.build(tmp_path / "fixture", "fixture-cpu", free_port())
    root, mirror = Path(fixture["root"]), Path(fixture["mirror"])
    initial = json.loads(Path(fixture["initialChannel"]).read_bytes())
    updated = json.loads(Path(fixture["updatedChannel"]).read_bytes())
    empty_catalog = tmp_path / "empty.lock"
    empty_catalog.write_text(Lockfile.of([]).record_json())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    catalog_flags = ("--catalog-lock", str(empty_catalog), "--artifacts", str(artifacts))
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text(
        HEALTHY_MANIFEST.replace('name = "healthy-nodes"', 'name = "healthy"')
        .replace("healthy_nodes:", "healthy:")
        .replace('requires = ["numpy>=1.26"]', "requires = []")
        .replace("[pack]\n", '[pack]\nversion = "1.0.0"\nengine = ">=1,<3"\n')
    )
    (pack / "healthy").mkdir()
    (pack / "healthy/__init__.py").write_text(HEALTHY_NODES)
    (pack / "pyproject.toml").write_text('[project]\nname="healthy"\nversion="1.0.0"\n')
    archive = tmp_path / "healthy-1.0.0-py3-none-any.whl"
    build_pack_archive(pack, archive)
    payload = archive.read_bytes()
    entry = LockedPack(
        "healthy", "1.0.0", artifact_digest(payload), "dinkster", ("healthy",), "registry:official"
    )
    lock_path = tmp_path / "healthy.pack-lock.json"
    PackLock("healthy", "fixture-cpu", ">=3.12", (), (), ()).write(lock_path)
    lock = lock_path.read_bytes()
    lock_url = "locks/" + artifact_digest(lock).replace(":", "/") + ".json"
    release = {
        **_static_release(entry, payload),
        "engine": ">=1,<3",
        "installLocks": {
            "fixture-cpu": {
                "baseId": initial["base"]["fixture-cpu"]["id"],
                "artifactDigest": entry.artifact_digest,
                "digest": artifact_digest(lock),
                "size": len(lock),
                "url": lock_url,
            }
        },
    }
    registry_files = {
        **_static_indexes([release]),
        "/static/" + lock_url: lock,
        "/static/" + release["wheels"][0]["url"]: payload,
    }
    server = _serve_artifacts(registry_files, [])
    try:
        setup = fixture["commands"]["setup"]
        code, result = _run(root, *setup[setup.index("setup") : -1], *catalog_flags, timeout=120)
        assert code == 0 and result["current"] == 1, result
        assert _run(root, "stop", "--yes")[0] == 0
        shutil.copyfile(fixture["updatedChannel"], mirror / "channels/main.json")
        code, result = _run(root, "update", *catalog_flags, "--yes", timeout=120)
        assert code == 0 and result["current"] == 2, result
        assert _run(root, "stop", "--yes")[0] == 0
        (root / "registries.toml").write_text(
            "[registries.official]\n"
            f'endpoint="http://127.0.0.1:{server.server_port}"\ndefault=true\n'
        )
        review = tmp_path / "review"
        if pack_route == "command":
            store = Path((root / "store").read_text().strip())
            before = {
                path: content_digest(path)
                for directory in (root, store)
                for path in directory.rglob("*")
                if path.is_file()
            }
            registry_config = tmp_path / "external-registries.toml"
            registry_text = (root / "registries.toml").read_text()
            registry_config.write_text(registry_text)
            command = (
                "--root",
                str(root),
                "--registries",
                str(registry_config),
                "install",
                "healthy@1.0.0",
            )
            with monkeypatch.context() as terminal:
                terminal.setattr(sys, "stdin", io.StringIO())
                with pytest.raises(SystemExit):
                    run_cli(*command)
                refusal = capsys.readouterr()
                assert "--yes" in refusal.err
                assert "Pack payload upper bound:" in refusal.out
                assert "proposed generation: 3" in refusal.out
                assert "healthy: absent -> 1.0.0" in refusal.out
                assert "Source healthy: download required" in refusal.out
                terminal.setattr(sys.stdin, "isatty", lambda: True)
                terminal.setattr("builtins.input", lambda _: "no")
                run_cli(*command)
                assert "nothing changed" in capsys.readouterr().out
            assert {
                path: content_digest(path)
                for directory in (root, store)
                for path in directory.rglob("*")
                if path.is_file()
            } == before
            run_cli(*command, "--plan", str(review))
            capsys.readouterr()
            saved = json.loads(review.read_bytes())
            registry_config.write_text(registry_text + "\n# changed configuration\n")
            with pytest.raises(SystemExit):
                run_cli("--root", str(root), "apply", str(review))
            assert "stale" in capsys.readouterr().err
            registry_config.write_text(registry_text)
            tampered = tmp_path / "tampered.json"
            malformed = {**saved, "registryArguments": ["--help"]}
            tampered.write_text(json.dumps(malformed))
            with pytest.raises(SystemExit):
                run_cli("--root", str(root), "apply", str(tampered))
            assert "invalid pack plan" in capsys.readouterr().err
            saved["review"]["current"] = 999
            tampered.write_text(json.dumps(saved))
            with pytest.raises(SystemExit):
                run_cli("--root", str(root), "apply", str(tampered))
            assert "stale" in capsys.readouterr().err
            run_cli("--root", str(root), "apply", str(review))
            assert "activated generation 3" in capsys.readouterr().out
        else:
            code, planned = _run(root, "packs-plan", "healthy@1.0.0", "--output", str(review))
            assert code == 0, planned
            assert len(history(root)) == 2
            code, result = _run(
                root,
                "packs-apply",
                "--input",
                str(review),
                "--approval",
                planned["approval"],
                timeout=120,
            )
            assert code == 0 and result["current"] == 3, result
        code, listed = _run(root, "history")
        assert code == 0, listed
        records = listed["history"]
        assert [row["id"] for row in records] == [1, 2, 3]
        assert [row["generation"] for row in records] == [1, 2, 3]
        assert [row["reason"] for row in records] == ["setup", "update", "update"]
        snapshots = [row["snapshot"] for row in records]
        assert [row["engine"]["release"] for row in snapshots] == [
            initial["engine"]["release"],
            updated["engine"]["release"],
            updated["engine"]["release"],
        ]
        assert snapshots[0]["base"] == {
            "id": initial["base"]["fixture-cpu"]["id"],
            "variant": "fixture-cpu",
        }
        assert all(row["base"] == snapshots[0]["base"] for row in snapshots)
        assert all(row["tools"] == {"version": version("dinkster-supervisor")} for row in snapshots)
        assert all(row["inProcessPacks"] == [] for row in snapshots)
        assert all(
            json.loads(row["metadata"]["dinkster.lock"])["packs"] == [] for row in snapshots[:2]
        )
        assert Lockfile.from_record_json(snapshots[2]["metadata"]["dinkster.lock"]) == Lockfile.of(
            [entry]
        )
        assert (
            snapshots[2]["metadata"]["packs/healthy/pack-lock.json"].encode("utf-8")
            == (root / "generations/3/packs/healthy/pack-lock.json").read_bytes()
        )
        if pack_route == "command":
            from dinkster_supervisor.environment_generations import restore_pack_sources

            transport = tmp_path / "restored-archives"
            restore_pack_sources(
                root / "generations/3",
                catalog_lock=root / "generations/3/dinkster.lock",
                artifacts=transport,
                project=root,
                store=tmp_path / "empty-store",
            )
            assert pack_archive_path(transport, entry.artifact_digest).read_bytes() == payload
            registry_files["/static/" + release["wheels"][0]["url"]] = b"x" * len(payload)
            unavailable = replace(entry, pack="missing-pack", artifact_digest="blake3:" + "a" * 64)
            failed_lock = tmp_path / "missing-sources.lock"
            failed_lock.write_text(Lockfile.of([entry, unavailable]).record_json())
            with pytest.raises(EnvironmentError) as missing_sources:
                restore_pack_sources(
                    root / "generations/3",
                    catalog_lock=failed_lock,
                    artifacts=tmp_path / "failed-archives",
                    project=root,
                    store=tmp_path / "empty-store",
                )
            assert entry.artifact_digest in str(missing_sources.value)
            assert unavailable.artifact_digest in str(missing_sources.value)
            assert not (tmp_path / "failed-archives").exists()
            registry_files["/static/" + release["wheels"][0]["url"]] = payload
        receipt_name = "packs/healthy/installed-wheels.json"
        assert (
            snapshots[2]["metadata"][receipt_name].encode("utf-8")
            == (root / "generations/3" / receipt_name).read_bytes()
        )
        generation = Generation.read(root / "generations/3/generation.json")
        receipt = root / "generations/3" / receipt_name
        assert dict(generation.files)[receipt_name] == content_digest(receipt)
        original_receipt = receipt.read_bytes()
        receipt.write_bytes(b"{}\n")
        with pytest.raises(EnvironmentError, match="intact software metadata"):
            capture_software(receipt.parent.parent.parent, inspect_wheel_tags=False)
        receipt.write_bytes(original_receipt)
        assert "packs/healthy/pack-lock.json" not in snapshots[0]["metadata"]
        code, shown = _run(root, "history", "show", "2")
        assert code == 0 and shown["entry"] == records[1], shown
        assert {"field": "engine.version", "before": "1.0", "after": "2.0"} in records[1]["diff"]
        assert {
            "field": "metadata.dinkster.lock",
            "before": snapshots[1]["metadata"]["dinkster.lock"],
            "after": snapshots[2]["metadata"]["dinkster.lock"],
        } in records[2]["diff"]
        preserved = {path: path.read_bytes() for path in (root / "history").glob("*.json")}
        assert _run(root, "stop", "--yes")[0] == 0
        code, result = _run(root, "restore", "1", "--yes", timeout=120)
        assert code == 0 and result["current"] == 4 and result["exact"] is True, result
        code, listed = _run(root, "history")
        assert code == 0 and len(listed["history"]) == 4, listed
        restored = listed["history"][-1]
        assert listed["history"][:3] == records
        assert restored["generation"] == 4 and restored["restoredFrom"] == 1
        assert restored["reason"] == "restore" and restored["exact"] is True
        assert restored["substitutions"] == []
        assert restored["snapshot"] == snapshots[0]
        assert restored["recordedTools"] == snapshots[0]["tools"]
        assert (root / "current").read_text() == "4\n"
        assert not (root / "generations/4/packs/healthy").exists()
        materialized = Generation.read(root / "generations/4/generation.json")
        assert materialized.in_process_packs == ()
        assert materialized.engine_release == initial["engine"]["release"]
        assert materialized.engine_version == "1.0"
        assert materialized.base == initial["base"]["fixture-cpu"]["id"]
        assert materialized.variant == "fixture-cpu"
        assert materialized.constraints == snapshots[0]["constraints"]
        for name, content in snapshots[0]["metadata"].items():
            assert (root / "generations/4" / name).read_bytes() == content.encode("utf-8")
        assert all(path.read_bytes() == payload for path, payload in preserved.items())
    finally:
        _run(root, "stop", "--yes")
        server.shutdown()


@pytest.mark.parametrize("existing_history", [False, True])
def test_legacy_import_preserves_activation_selection_and_refuses_conflicting_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_history: bool
):
    (tmp_path / "journal").mkdir()
    for operation, previous, number in ((10, None, 7), (20, 7, 2), (30, 2, 9)):
        tree = _generation(tmp_path, number).parent
        python = interpreter(tree)
        python.parent.mkdir(exist_ok=True)
        python.write_bytes(b"sealed interpreter fixture")
        record = Generation.read(tree / "generation.json")
        files = (*record.files, (python.relative_to(tree).as_posix(), content_digest(python)))
        if number == 7:
            lock = tree / "packs/example/pack-lock.json"
            lock.parent.mkdir(parents=True)
            lock.write_bytes(b"{}\n")
            files = (*files, (lock.relative_to(tree).as_posix(), content_digest(lock)))
        replace(
            record,
            engine_version=f"0.3.{number}",
            files=files,
        ).write(tree / "generation.json")
        entry = None
        if existing_history and number != 7:
            entry = prepare_entry(tmp_path, tree, operation=str(operation), reason="update")
            publish_entry(tmp_path, entry)
        Journal(
            "setup" if previous is None else "update",
            previous,
            number,
            "complete",
            history_entry=entry,
        ).write(tmp_path / "journal" / f"{operation}.json")
    set_current(tmp_path, 9)
    old_entries = {path: path.read_bytes() for path in (tmp_path / "history").glob("*.json")}
    with writer_lock(tmp_path), monkeypatch.context() as patch:

        def forbidden_probe(*args, **kwargs):
            raise AssertionError("legacy import must not execute the retained interpreter")

        patch.setattr("dinkster_supervisor.environment_history.subprocess.run", forbidden_probe)
        imported = import_generation(tmp_path, 7)
        assert import_generation(tmp_path, 7) == imported
    assert imported["kind"] == "import" and imported["reason"] == "legacy-import"
    assert imported["generation"] == 7 and imported["operation"] == "10"
    assert cast("dict", imported["snapshot"])["tools"] == {"version": "unknown"}
    assert cast("dict", imported["snapshot"])["host"] == {
        "system": "unknown",
        "machine": "unknown",
        "python": "unknown",
    }
    assert imported["provenance"] == {
        "journal": "10",
        "digest": content_digest(tmp_path / "journal/10.json"),
        "operation": "setup",
        "previous": None,
        "target": 7,
        "stage": "complete",
    }
    assert [entry["generation"] for entry in activation_entries(tmp_path)] == (
        [2, 9] if existing_history else []
    )
    assert (tmp_path / "current").read_bytes() == b"9\n"
    assert all(path.read_bytes() == payload for path, payload in old_entries.items())
    if existing_history:
        assert (
            prepare_entry(tmp_path, tmp_path / "generations/9", operation="40", reason="update")[
                "diff"
            ]
            == []
        )
    preserved = {path: path.read_bytes() for path in (tmp_path / "history").glob("*.json")}
    seal = tmp_path / "generations/7/generation.json"
    original = seal.read_bytes()
    replace(Generation.read(seal), engine_version="9.9").write(seal)
    with (
        writer_lock(tmp_path),
        pytest.raises(EnvironmentError, match="conflicts with recorded software"),
    ):
        import_generation(tmp_path, 7)
    seal.write_bytes(original)
    Journal("update", 7, 2, "planned").write(tmp_path / "journal/40.json")
    with (
        writer_lock(tmp_path),
        pytest.raises(EnvironmentError, match="completed operation recovery"),
    ):
        import_generation(tmp_path, 7)
    (tmp_path / "journal/40.json").unlink()
    Journal("update", 9, 7, "complete").write(tmp_path / "journal/40.json")
    with (
        writer_lock(tmp_path),
        pytest.raises(EnvironmentError, match="unambiguous completed ancestry"),
    ):
        import_generation(tmp_path, 7)
    (tmp_path / "journal/40.json").unlink()
    entry = prepare_entry(tmp_path, tmp_path / "generations/9", operation="40", reason="update")
    Journal("update", 2, 9, "complete", history_entry=entry).write(tmp_path / "journal/40.json")
    with (
        writer_lock(tmp_path),
        pytest.raises(EnvironmentError, match="completed history publication"),
    ):
        import_generation(tmp_path, 7)
    (tmp_path / "journal/40.json").unlink()
    with writer_lock(tmp_path), pytest.raises(EnvironmentError, match="disconnected"):
        import_generation(tmp_path, 5)
    python = interpreter(tmp_path / "generations/7")
    payload = python.read_bytes()
    python.unlink()
    with (
        writer_lock(tmp_path),
        pytest.raises(EnvironmentError, match="sealed generation interpreter"),
    ):
        import_generation(tmp_path, 7)
    python.write_bytes(payload)
    setup = tmp_path / "journal/10.json"
    original_journal = setup.read_bytes()
    setup.write_bytes(original_journal + b"\n")
    with writer_lock(tmp_path), pytest.raises(EnvironmentError, match="import provenance"):
        import_generation(tmp_path, 7)
    setup.write_bytes(original_journal)
    unexpected = tmp_path / "generations/7/unsealed.py"
    unexpected.write_bytes(b"unsealed software")
    with writer_lock(tmp_path), pytest.raises(EnvironmentError, match="intact generation seal"):
        import_generation(tmp_path, 7)
    unexpected.unlink()
    (tmp_path / "generations/7/constraints.txt").write_bytes(b"changed")
    with writer_lock(tmp_path), pytest.raises(EnvironmentError, match="intact generation seal"):
        import_generation(tmp_path, 7)
    assert all(path.read_bytes() == payload for path, payload in preserved.items())


def test_imported_software_restores_after_pruning_without_becoming_rollback_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tests.fixtures.environments.build import build

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    root, mirror = Path(fixture["root"]), Path(fixture["mirror"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        legacy_snapshot = history(root)[0]["snapshot"]
        assert _run(root, "stop", "--yes")[0] == 0
        assert _run(root, "cleanup", "--automatic", "off", "--yes")[0] == 0
        setup_journal = next((root / "journal").glob("*.json"))
        replace(Journal.read(setup_journal), history_entry=None).write(setup_journal)
        (root / "history/00000000000000000001.json").unlink()
        shutil.copyfile(fixture["updatedChannel"], mirror / "channels/main.json")
        code, result = _run(root, "update", "--yes")
        assert code == 0 and result["current"] == 2, result
        assert _run(root, "stop", "--yes")[0] == 0
        executable_release(
            mirror, json.loads(Path(fixture["baseManifest"]).read_bytes()), "3.0", 73
        )
        code, result = _run(root, "update", "--yes")
        assert code == 0 and result["current"] == 3, result
        assert _run(root, "stop", "--yes")[0] == 0
        preserved = {path: path.read_bytes() for path in (root / "history").glob("*.json")}
        for flags in (("--plan",), ()):
            code, result = _run(root, "import", "1", *flags)
            assert code == 0 and result["status"] == "planned", result
            assert len(history(root)) == 2
        with writer_lock(root):
            code, result = _run(root, "import", "1", "--yes")
            assert code == 1 and "another environment writer" in result["error"], result
        # A confirmed import recovers an unpublished completed activation first.
        (root / "history/00000000000000000002.json").unlink()
        code, result = _run(root, "import", "1", "--yes")
        assert code == 0 and result["status"] == "applied", result
        imported = result["entry"]
        code, result = _run(root, "import", "1", "--yes")
        assert code == 0 and result["entry"] == imported, result
        assert imported["id"] == 3 and imported["kind"] == "import"
        assert (root / "current").read_text() == "3\n"
        assert all(path.read_bytes() == payload for path, payload in preserved.items())
        assert _run(root, "cleanup", "--automatic", "on", "--yes")[0] == 0
        code, result = _run(root, "rollback", "--yes")
        assert code == 0 and result["current"] == 4, result
        assert history(root)[-1]["restoredFrom"] == 1
        assert Generation.read(root / "generations/4/generation.json").engine_version == "2.0"
        assert _run(root, "stop", "--yes")[0] == 0
        missing_snapshot = cast("dict", history(root)[1]["snapshot"])
        manifest = json.loads(missing_snapshot["metadata"]["engine-manifest.json"])
        shutil.rmtree(mirror / "engine" / missing_snapshot["engine"]["release"][7:])
        for artifact in manifest["artifacts"]:
            if artifact["name"].endswith(".whl"):
                cached = object_path(
                    Path(fixture["store"]) / "wheels", "sha256:" + artifact["sha256"]
                )
                if cached.exists():
                    shutil.rmtree(cached)
        code, proposal = _run(root, "restore", "2", "--plan")
        assert code == 0 and proposal["status"] == "approval-required", proposal
        assert any(
            candidate["snapshot"] == imported["snapshot"]
            for candidate in proposal["restorePlan"]["alternatives"]
        )
        assert sorted(path.name for path in (root / "generations").iterdir()) == ["3", "4"]
        code, shown = _run(root, "history", "show", "3")
        assert code == 0 and shown["entry"] == imported
        journal_bytes = setup_journal.read_bytes()
        preserved_imports = {path: path.read_bytes() for path in (root / "history").glob("*.json")}
        for mutation in (journal_bytes + b"\n", None):
            if mutation is None:
                setup_journal.unlink()
            else:
                setup_journal.write_bytes(mutation)
            code, refused = _run(root, "restore", "3", "--yes")
            assert code == 1 and "provenance" in refused["error"], refused
            assert (root / "current").read_text() == "4\n"
            assert all(path.read_bytes() == payload for path, payload in preserved_imports.items())
            code, proposal = _run(root, "restore", "2", "--plan")
            assert code == 0 and proposal["status"] == "approval-required", proposal
            assert all(
                candidate["snapshot"] != imported["snapshot"]
                for candidate in proposal["restorePlan"]["alternatives"]
            )
            setup_journal.write_bytes(journal_bytes)
        code, result = _run(root, "restore", "3", "--yes")
        assert code == 0 and result["current"] == 5 and result["exact"] is True, result
        restored = history(root)[-1]
        assert restored["restoredFrom"] == 3
        assert restored["recordedTools"] == {"version": "unknown"}
        assert {
            key: value
            for key, value in cast("dict", restored["snapshot"]).items()
            if key not in {"tools", "host"}
        } == {
            key: value
            for key, value in cast("dict", legacy_snapshot).items()
            if key not in {"tools", "host"}
        }
        assert all(path.read_bytes() == payload for path, payload in preserved.items())
        assert [entry["generation"] for entry in activation_entries(root)] == [2, 3, 4, 5]
    finally:
        _run(root, "stop", "--yes")


def test_standalone_adoption_builds_separate_history_without_touching_old_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_registry import Lockfile

    from tests.fixtures.environments.build import build

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    source = tmp_path / "standalone"
    (source / "generations").mkdir(parents=True)
    (source / "current").write_text("7\n")
    (source / "generations/7.json").write_text(Lockfile().record_json())
    for folder in ("workflows", "models", "outputs", "mounts"):
        (source / folder).mkdir()
        (source / folder / "keep.bin").write_bytes(b"never adopt or modify user data")
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    channel = json.loads(Path(fixture["initialChannel"]).read_bytes())
    release, base = channel["engine"]["release"], channel["base"]["fixture-cpu"]["id"]
    arguments = (
        "setup",
        "--adopt-from",
        str(source),
        "--variant",
        "fixture-cpu",
        "--mirror",
        str(mirror),
        "--channel",
        "main",
        "--data",
        str(store),
        "--engine-release",
        release,
        "--base-id",
        base,
        "--port",
        str(fixture["port"]),
    )
    for flags in ((), ("--plan",)):
        code, planned = _run(root, *arguments, *flags)
        assert code == 0 and planned["status"] == "planned", planned
        assert planned["adoption"]["generation"] == 7
        assert planned["adoption"]["previousEngine"] == "not recorded by standalone installer"
        assert not root.exists() and not store.exists()
        assert not (tmp_path / "user-data").exists()
    code, mismatch = _run(root, *arguments, "--base-id", "sha256:" + "f" * 64, "--yes")
    assert code == 1 and "does not match" in mismatch["error"], mismatch
    assert not root.exists() and not store.exists()
    code, diagnostic = _run(source, "history")
    assert code == 1 and "--adopt-from" in diagnostic["error"]
    code, applied = _run(root, *arguments, "--yes")
    assert code == 0 and applied["status"] == "applied", applied
    try:
        entries = history(root)
        assert len(entries) == 1
        snapshot = cast("dict", entries[0]["snapshot"])
        assert snapshot["engine"]["release"] == release
        assert snapshot["base"]["id"] == base
        assert json.loads(snapshot["metadata"]["dinkster.lock"])["packs"] == []
        assert {
            path.relative_to(source): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        } == before
        assert all(
            not (root / name).exists() for name in ("workflows", "models", "outputs", "mounts")
        )
    finally:
        assert _run(root, "stop", "--yes")[0] == 0


@pytest.mark.parametrize(
    ("replace_base", "replace_engine", "reacquire", "mutation"),
    [
        (False, False, False, "update"),
        (True, False, False, "update"),
        (False, True, False, "update"),
        (True, True, False, "update"),
        (False, False, True, "update"),
        (False, False, False, "pack-install"),
    ],
    ids=[
        "exact",
        "missing-base",
        "missing-engine",
        "missing-engine-and-base",
        "reacquired-source",
        "pack-install",
    ],
)
def test_adoption_materializes_pack_dependencies_and_restores_after_pruning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_base: bool,
    replace_engine: bool,
    reacquire: bool,
    mutation: str,
):
    from dinkster_registry import LockedPack, Lockfile, artifact_digest, build_artifact
    from dinkster_supervisor.environment_state import interpreter
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.pack_environment import venv_python
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES
    from test_environment_generations import _wheel, _workspace_tree
    from test_installer import _serve_artifacts, _static_indexes, _static_release

    from dinkster.pack_archive import build_pack_archive
    from tests.fixtures.environments import build as publisher

    original_wheel, encode = publisher.wheel, publisher.canonical_json

    def published_wheel(*args, **kwargs):
        return original_wheel(*args, **kwargs, pack_tools=True)

    def canonical_json(document: dict) -> bytes:
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        return encode(document)

    monkeypatch.setattr(publisher, "wheel", published_wheel)
    monkeypatch.setattr(publisher, "standalone_tree", _workspace_tree)
    monkeypatch.setattr(publisher, "canonical_json", canonical_json)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = publisher.build(tmp_path / "fixture", "fixture-cpu", free_port())
    source = tmp_path / "standalone"
    archives = source / "artifacts"
    archives.mkdir(parents=True)
    (source / "generations").mkdir()
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    updated_artifacts = tmp_path / "updated-artifacts"
    updated_artifacts.mkdir()
    payload, digest = _wheel(dependencies, "fixture_adopt", "1.0", 87)
    entries, locks = [], []
    for name in ("healthy", "other", "extra"):
        pack = tmp_path / name
        pack.mkdir()
        (pack / "dinkster-pack.toml").write_text(
            HEALTHY_MANIFEST.replace("healthy-nodes", "healthy")
            .replace('publisher = "dinkster"\n', "")
            .replace("healthy", name)
            .replace(f"{name}_nodes:", f"{name}:")
            .replace("numpy>=1.26", "fixture-adopt==1.0")
            .replace(
                "[pack]\n",
                '[pack]\nversion = "1.0.0"\nengine = ">=0.0.1,<3.0"\n'
                + ('publisher="dinkster"\n' if name == "extra" else ""),
            )
        )
        (pack / name).mkdir()
        (pack / name / "__init__.py").write_text(HEALTHY_NODES.replace("healthy", name))
        staging = updated_artifacts / "staging.zip"
        if name == "extra":
            (pack / "pyproject.toml").write_text('[project]\nname="extra"\nversion="1.0.0"\n')
            staging = updated_artifacts / "extra-1.0.0-py3-none-any.whl"
            build_pack_archive(pack, staging)
            identity = artifact_digest(staging.read_bytes())
        else:
            identity = build_artifact(pack, staging)
        retained = updated_artifacts / f"{identity[7:]}.zip"
        staging.rename(retained)
        if name != "extra":
            shutil.copyfile(retained, archives / retained.name)
        entries.append(
            LockedPack(
                name,
                "1.0.0",
                identity,
                "dinkster" if name == "extra" else "local",
                (name,),
                f"local:{pack}",
            )
        )
        path = dependencies / f"{name}.pack-lock.json"
        PackLock(
            name,
            "fixture-cpu",
            ">=3.12",
            ("fixture-adopt==1.0",),
            (),
            (
                LockedDistribution(
                    "fixture-adopt",
                    "1.0",
                    (LockedWheel(payload.name, payload.stat().st_size, digest, payload.as_uri()),),
                ),
            ),
        ).write(path)
        locks.extend(("--pack-lock", str(path)))
    (source / "current").write_text("3\n")
    source_lock = Lockfile.of(entries[:2]).record_json()
    (source / "generations/3.json").write_text(source_lock)
    updated_lock = tmp_path / "updated.lock"
    updated_lock.write_text(Lockfile.of(entries).record_json())
    (source / "generations/3.hosting.json").write_text(
        json.dumps({"format": "dinkster.hosting/1", "inProcess": ["healthy"]})
    )
    for name in ("workflows", "models", "outputs", "mounts"):
        (source / name).mkdir()
        (source / name / "keep.bin").write_bytes(b"standalone user data")
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    channel = json.loads(Path(fixture["initialChannel"]).read_bytes())
    extra = replace(entries[-1], publisher="dinkster", source="registry")
    extra_lock = (dependencies / "extra.pack-lock.json").read_bytes()
    extra_archive = (updated_artifacts / f"{extra.artifact_digest[7:]}.zip").read_bytes()
    lock_url = "locks/" + artifact_digest(extra_lock).replace(":", "/") + ".json"
    release = {
        **_static_release(extra, extra_archive),
        "engine": ">=0.0.1,<3.0",
        "installLocks": {
            "fixture-cpu": {
                "baseId": channel["base"]["fixture-cpu"]["id"],
                "artifactDigest": extra.artifact_digest,
                "digest": artifact_digest(extra_lock),
                "size": len(extra_lock),
                "url": lock_url,
            }
        },
    }
    outer_wheel = release["wheels"][0]
    registry = _serve_artifacts(
        {
            **_static_indexes([release]),
            "/static/" + lock_url: extra_lock,
            "/static/" + outer_wheel["url"]: extra_archive,
        },
        [],
    )
    code, result = _run(
        root,
        "setup",
        "--adopt-from",
        str(source),
        "--engine-release",
        channel["engine"]["release"],
        "--base-id",
        channel["base"]["fixture-cpu"]["id"],
        "--variant",
        "fixture-cpu",
        "--mirror",
        str(mirror),
        "--data",
        str(store),
        "--channel",
        "main",
        "--port",
        str(fixture["port"]),
        *locks[:4],
        "--yes",
        timeout=120,
    )
    try:
        assert code == 0 and result["current"] == 1, result.get("error", result)
        first = history(root)[0]
        snapshot = cast("dict", first["snapshot"])
        replacement_id = None
        chosen = None
        expected_engine = snapshot["engine"]
        assert (
            "pack-sources/healthy/dinkster-pack.toml" in cast("dict", first["snapshot"])["metadata"]
        )
        assert (
            "pack-sources/healthy/healthy/__init__.py"
            not in cast("dict", first["snapshot"])["metadata"]
        )
        for number in (1, 3):
            generation = root / "generations" / str(number)
            assert json.loads((generation / "dinkster.lock").read_text()) == json.loads(source_lock)
            assert Generation.read(generation / "generation.json").in_process_packs == ("healthy",)
            assert not (generation / "packs/extra").exists()
            for python in (interpreter(generation), venv_python(generation / "packs/other")):
                assert (
                    subprocess.check_output(
                        [
                            str(python),
                            "-I",
                            "-B",
                            "-c",
                            "import fixture_adopt; print(fixture_adopt.VALUE)",
                        ],
                        text=True,
                    ).strip()
                    == "87"
                )
            if number == 1:
                assert _run(root, "stop", "--yes")[0] == 0
                (mirror / "channels/main.json").write_bytes(
                    Path(fixture["updatedChannel"]).read_bytes()
                )
                if mutation == "pack-install":
                    (root / "registries.toml").write_text(
                        "[registries.official]\n"
                        f'endpoint="http://127.0.0.1:{registry.server_port}"\ndefault=true\n'
                    )
                    review = tmp_path / "review"
                    journals = sorted((root / "journal").glob("*"))
                    code, planned = _run(root, "packs-plan", "extra@1.0.0", "--output", str(review))
                    assert code == 0, planned.get("error", planned)
                    assert sorted((root / "journal").glob("*")) == journals
                    pack_approval = planned["approval"]
                    for folder in ("workflows", "models", "outputs", "user"):
                        (root / folder).mkdir(exist_ok=True)
                        (root / folder / "sentinel.bin").write_bytes(
                            folder.encode() + b"\x00unchanged"
                        )
                    data_before = {
                        folder: (root / folder / "sentinel.bin").read_bytes()
                        for folder in ("workflows", "models", "outputs", "user")
                    }
                    retained_locks = {
                        path.parent.name: path.read_bytes()
                        for path in (root / "generations/1/packs").glob("*/pack-lock.json")
                    }
                    settings = root / "project.json"
                    saved = settings.read_bytes()
                    settings.write_bytes(saved + b"\n")
                    code, stale = _run(
                        root, "packs-apply", "--input", str(review), "--approval", pack_approval
                    )
                    assert code != 0 and "changed after planning" in stale["error"]
                    assert sorted((root / "journal").glob("*")) == journals
                    assert not (root / "generations/2").exists()
                    settings.write_bytes(saved)
                    unreviewed = review / "locks/unreviewed.json"
                    unreviewed.write_bytes(extra_lock)
                    code, tampered = _run(
                        root, "packs-apply", "--input", str(review), "--approval", pack_approval
                    )
                    assert code != 0 and "reviewed pack inputs changed" in tampered["error"]
                    assert sorted((root / "journal").glob("*")) == journals
                    unreviewed.unlink()
                    code, updated = _run(
                        root, "packs-apply", "--input", str(review), "--approval", pack_approval
                    )
                    assert code == 0, updated.get("error", updated)
                    assert {
                        folder: (root / folder / "sentinel.bin").read_bytes()
                        for folder in data_before
                    } == data_before
                    for name, payload in retained_locks.items():
                        assert (
                            root / "generations/2/packs" / name / "pack-lock.json"
                        ).read_bytes() == payload
                    endpoint = f"http://127.0.0.1:{registry.server_port}"
                    assert PackLock.read(
                        root / "generations/2/packs/extra/pack-lock.json"
                    ) == replace(
                        PackLock.read(dependencies / "extra.pack-lock.json"),
                        indexes=(endpoint + "/static/fixture-cpu/simple/",),
                        packages=(
                            *PackLock.read(dependencies / "extra.pack-lock.json").packages,
                            LockedDistribution(
                                "extra",
                                "1.0.0",
                                (
                                    LockedWheel(
                                        outer_wheel["filename"],
                                        len(extra_archive),
                                        hashlib.sha256(extra_archive).hexdigest(),
                                        endpoint + "/static/" + outer_wheel["url"],
                                    ),
                                ),
                            ),
                        ),
                    )
                    code_replay, _replayed = _run(
                        root, "packs-apply", "--input", str(review), "--approval", pack_approval
                    )
                    assert code_replay != 0 and not (root / "generations/3").exists()
                else:
                    code, updated = _run(
                        root,
                        "update",
                        "--catalog-lock",
                        str(updated_lock),
                        "--artifacts",
                        str(updated_artifacts),
                        *locks,
                        "--yes",
                        timeout=120,
                    )
                assert code == 0 and updated["current"] == 2, updated.get("error", updated)
                assert (root / "generations/2/packs/extra/pack-lock.json").is_file()
                assert _run(root, "stop", "--yes")[0] == 0
                shutil.rmtree(root / "generations/1")
                original_release = cast("dict", first["snapshot"])["engine"]["release"]
                current_release = Generation.read(
                    root / "generations/2/generation.json"
                ).engine_release
                assert (current_release == original_release) == (mutation == "pack-install")
                for entry in entries[:2]:
                    filename = f"{entry.artifact_digest[7:]}.zip"
                    assert (
                        pack_archive_path(
                            object_path(store / "engines", current_release) / "artifacts",
                            entry.artifact_digest,
                        ).read_bytes()
                        == (archives / filename).read_bytes()
                    )
                original_archives = object_path(store / "engines", original_release) / "artifacts"
                if mutation != "pack-install":
                    shutil.rmtree(original_archives)
                corrupt_cache = store / "engines" / ("0" * 64) / "artifacts"
                corrupt_cache.mkdir(parents=True)
                for entry in entries[:2]:
                    (corrupt_cache / f"{entry.artifact_digest[7:]}.zip").write_bytes(
                        b"not the digest-pinned pack archive"
                    )
                if reacquire:
                    for entry in entries[:2]:
                        pack_archive_path(
                            object_path(store / "engines", current_release) / "artifacts",
                            entry.artifact_digest,
                        ).unlink()
                    preserved_sources = {
                        path: path.read_bytes()
                        for parent in (root, store)
                        for path in parent.rglob("*")
                        if path.is_file() and path.name != ".lock"
                    }
                    code, available = _run(root, "restore", "1", "--plan")
                    assert code == 0 and available["restorePlan"]["exact"], available
                    changed = tmp_path / "healthy/healthy/__init__.py"
                    original_nodes = changed.read_bytes()
                    for content in (original_nodes + b"\n# changed source bytes\n", None):
                        if content is None:
                            changed.unlink()
                        else:
                            changed.write_bytes(content)
                        code, unavailable = _run(root, "restore", "1", "--yes")
                        assert code == 0 and unavailable["status"] == "approval-required", (
                            unavailable
                        )
                        assert unavailable["restorePlan"]["missing"] == [
                            {
                                "kind": "pack-source",
                                "identity": entries[0].artifact_digest,
                                "artifact": f"{entries[0].artifact_digest[7:]}.zip",
                                "pack": "healthy",
                                "source": f"local:{tmp_path / 'healthy'}",
                            }
                        ]
                        assert not unavailable["restorePlan"]["exact"]
                    changed.write_bytes(original_nodes)
                    assert {
                        path: path.read_bytes()
                        for parent in (root, store)
                        for path in parent.rglob("*")
                        if path.is_file() and path.name != ".lock"
                    } == preserved_sources
                approval: tuple[str, ...] = ("--yes",)
                if replace_base:
                    snapshot = cast("dict", first["snapshot"])
                    original = json.loads(snapshot["metadata"]["base-manifest.json"])
                    base_id = snapshot["base"]["id"]
                    published = mirror / "base/fixture-cpu" / base_id[7:]
                    index = []
                    for compatible in (False, True):
                        base = json.loads(json.dumps(original))
                        base["interpreter"]["release"] = (
                            "compatible-replacement" if compatible else "conflicting-replacement"
                        )
                        if not compatible:
                            conflicting, sha = _wheel(dependencies, "fixture_adopt", "2.0", 99)
                            base["wheels"].append(
                                {
                                    "name": "fixture-adopt",
                                    "version": "2.0",
                                    "filename": conflicting.name,
                                    "sha256": sha,
                                    "size": conflicting.stat().st_size,
                                }
                            )
                            base["replaces"] = sorted([*base["replaces"], "fixture-adopt"])
                        identity_fields = {
                            key: value
                            for key, value in base.items()
                            if key not in {"id", "signature"}
                        }
                        identity_fields["archive"] = {"format": "tar+zstd"}
                        identity = hashlib.sha256(canonical_json(identity_fields)).hexdigest()
                        base["id"] = "sha256:" + identity
                        destination = published.parent / identity
                        shutil.copytree(published, destination)
                        (destination / "manifest.json").write_bytes(canonical_json(base))
                        index.append({"id": base["id"], "path": f"base/fixture-cpu/{identity}"})
                        if compatible:
                            replacement_id = base["id"]
                    (published.parent / "index.json").write_text(json.dumps(index))
                    shutil.rmtree(published)
                    shutil.rmtree(object_path(store / "bases", base_id))
                    shutil.rmtree(object_path(store / "downloads", base_id))
                if replace_engine:
                    snapshot = cast("dict", first["snapshot"])
                    base = json.loads(snapshot["metadata"]["base-manifest.json"])
                    channel_path = mirror / "channels/main.json"
                    index = []
                    for version in ("2.5", "3.0"):
                        publisher.executable_release(mirror, base, version, 73)
                        selected = json.loads(channel_path.read_bytes())["engine"]
                        index.append(
                            {
                                "sha256": selected["release"][7:],
                                "path": f"engine/{selected['release'][7:]}",
                            }
                        )
                        if version == "2.5":
                            expected_engine = {key: selected[key] for key in ("version", "release")}
                    (mirror / "engine/index.json").write_text(json.dumps(index))
                    original = snapshot["engine"]["release"]
                    shutil.rmtree(mirror / "engine" / original[7:])
                    for row in json.loads(snapshot["metadata"]["engine-manifest.json"])[
                        "artifacts"
                    ]:
                        if row["name"].endswith(".whl"):
                            shutil.rmtree(object_path(store / "wheels", "sha256:" + row["sha256"]))
                if replace_base or replace_engine:
                    preserved = {
                        path.relative_to(root): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file() and path.name != ".lock"
                    }
                    code, proposal = _run(root, "restore", "1", "--yes")
                    assert code == 0 and proposal["status"] == "approval-required", proposal
                    alternatives = proposal["restorePlan"]["alternatives"]
                    assert len(alternatives) == 1, alternatives
                    chosen = alternatives[0]
                    substitutions = chosen["substitutions"]
                    expected_fields = {"engine", "constraints"} if replace_engine else set()
                    if replace_base:
                        expected_fields.add("base")
                        assert next(row for row in substitutions if row["field"] == "base") == {
                            "field": "base",
                            "before": snapshot["base"],
                            "after": {"id": replacement_id, "variant": "fixture-cpu"},
                        }
                    assert {row["field"] for row in substitutions} == expected_fields
                    assert chosen["snapshot"]["engine"] == expected_engine
                    assert chosen["snapshot"]["base"] == (
                        {"id": replacement_id, "variant": "fixture-cpu"}
                        if replace_base
                        else snapshot["base"]
                    )
                    assert chosen["snapshot"]["inProcessPacks"] == ["healthy"]
                    assert {
                        path.relative_to(root): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file() and path.name != ".lock"
                    } == preserved
                    approval = ("--approve", chosen["approval"])
                code, restored = _run(root, "restore", "1", *approval, timeout=120)
                assert code == 0 and restored["current"] == 3, restored.get("error", restored)
        records = history(root)
        assert len(records) == 3 and records[0] == first
        assert records[1]["reason"] == "update"
        assert records[2]["restoredFrom"] == 1
        assert [entry["exact"] for entry in records] == [
            True,
            True,
            not (replace_base or replace_engine),
        ]
        if replace_base:
            assert replacement_id is not None and chosen is not None
            assert cast("dict", records[2]["snapshot"])["base"]["id"] == replacement_id
        if replace_base or replace_engine:
            assert chosen is not None
            assert records[2]["substitutions"] == chosen["substitutions"]
        assert cast("dict", records[2]["snapshot"])["engine"] == expected_engine
        assert {
            path.relative_to(source): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        } == before
    finally:
        _run(root, "stop", "--yes")
        registry.shutdown()


@pytest.mark.parametrize("shared_store", [False, True], ids=["private-store", "shared-store"])
@pytest.mark.parametrize("in_process", [False, True], ids=["isolated", "in-process"])
def test_adoption_plans_exact_pack_sources_and_refuses_unsafe_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shared_store: bool,
    in_process: bool,
):
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_supervisor import environment
    from dinkster_workers.pack_lock import LockedDistribution, PackLock
    from test_environment_artifacts import _mirror

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    mirror, source, target, store = (tmp_path / name for name in ("mirror", "old", "new", "store"))
    base, _, channel = _mirror(mirror)
    (source / "generations").mkdir(parents=True)
    (source / "artifacts").mkdir()
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "dinkster-pack.toml").write_text('[pack]\nname="example"\nnamespaces=[]\n')
    archive = tmp_path / "pack.zip"
    identity = build_artifact(pack, archive)
    retained = source / "artifacts" / f"{identity[7:]}.zip"
    shutil.copyfile(archive, retained)
    content = source
    if shared_store:
        shared = tmp_path / "standalone-content"
        shared.mkdir()
        (source / "artifacts").rename(shared / "artifacts")
        (source / "shared-store").write_text(str(shared) + "\n")
        retained = shared / "artifacts" / retained.name
        content = shared
        retained.unlink()
        readable = pack_archive_path(content / "artifacts", identity, label="example/1.2.3")
        readable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, readable)
        retained = readable
    locked = LockedPack("example", "1.2.3", identity, "local", (), "local:/unused")
    (source / "current").write_text("3\n")
    (source / "generations/3.json").write_text(Lockfile.of([locked]).record_json())
    hosting = source / "generations/3.hosting.json"
    hosting.write_text(
        json.dumps(
            {
                "format": "dinkster.hosting/1",
                "inProcess": ["example"] if in_process else [],
                "runtimePins": {"torch": "2.0"},
            }
        )
    )
    sites = [content / "venvs" / identity[7:] / "example/lib/python3.12/site-packages"]
    if shared_store:
        sites = [
            object_path(content / "venvs" / scope, identity, label="example/1.2.3")
            / "example/Lib/site-packages"
            for scope in ("cpu", "cuda")
        ]
    for index, site in enumerate(sites):
        site.mkdir(parents=True)
        for name, version in (
            ("Retained_Package", "2.0"),
            ("changed-package", f"1.{index}"),
            ("removed", "9.0"),
        ):
            info = site / f"{name}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n\n")
        (site / "do-not-execute.pth").write_text(
            f"import pathlib; pathlib.Path({str(tmp_path / 'executed')!r}).touch()\n"
        )
    dependency = tmp_path / "example.pack-lock.json"
    PackLock(
        "example",
        "fixture-cpu",
        ">=3.12",
        (),
        (),
        tuple(
            LockedDistribution(name, version, ())
            for name, version in (
                ("retained-package", "2.0"),
                ("changed-package", "2.4"),
                ("added", "4.1"),
            )
        ),
    ).write(dependency)
    args = (
        "setup",
        "--adopt-from",
        str(source),
        "--variant",
        "fixture-cpu",
        "--channel",
        "main",
        "--mirror",
        str(mirror),
        "--data",
        str(store),
        "--pack-lock",
        str(dependency),
        "--engine-release",
        channel["engine"]["release"],
        "--base-id",
        base["id"],
    )
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    code, plan = _run(target, *args, "--plan")
    assert code == 0, plan
    assert plan["adoption"]["packs"][0]["artifactDigest"] == identity
    assert plan["adoption"]["inProcessPacks"] == (["example"] if in_process else [])
    assert plan["dependencyLocks"][str(dependency)]["pack"] == "example"
    observed = plan["adoption"]["installedDependencies"]["example"]
    if in_process:
        assert observed == {
            "status": "ambient-not-recorded",
            "scopes": {},
            "runtimePins": {"torch": "2.0"},
        }
        assert plan["dependencyComparison"][0]["sourceStatus"] == "ambient-not-recorded"
    else:
        scopes = ["cpu", "cuda"] if shared_store else ["private"]
        assert set(observed["scopes"]) == set(scopes)
        for index, scope in enumerate(scopes):
            assert observed["scopes"][scope]["pins"] == [
                f"changed-package==1.{index}",
                "removed==9.0",
                "retained-package==2.0",
            ]
            comparison = next(row for row in plan["dependencyComparison"] if row["scope"] == scope)
            assert comparison["changes"] == [
                {"name": "added", "source": None, "targetLock": "4.1"},
                {"name": "changed-package", "source": f"1.{index}", "targetLock": "2.4"},
                {"name": "removed", "source": "9.0", "targetLock": None},
            ]
    assert not (tmp_path / "executed").exists()
    assert {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    } == before
    assert not target.exists() and not store.exists()
    for boundary in dict.fromkeys((source, content, content / "artifacts")):
        saved_boundary = boundary.with_name(boundary.name + "-unlinked")
        boundary.rename(saved_boundary)
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(boundary), str(saved_boundary)],
                    capture_output=True,
                    check=True,
                )
            else:
                boundary.symlink_to(saved_boundary, target_is_directory=True)
            code, refused = _run(target, *args, "--plan")
            assert code == 1 and "link" in refused["error"], refused
            assert not target.exists() and not store.exists()
        finally:
            if sys.platform == "win32":
                boundary.rmdir()
            else:
                boundary.unlink()
            saved_boundary.rename(boundary)
    linked_parent = tmp_path / "linked-parent"
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_parent), str(tmp_path)],
            capture_output=True,
            check=True,
        )
    else:
        linked_parent.symlink_to(tmp_path, target_is_directory=True)
    try:
        code, refused = _run(
            target, *args, "--adopt-from", str(linked_parent / source.name), "--plan"
        )
        assert code == 1 and "link" in refused["error"], refused
        if shared_store:
            pointer = source / "shared-store"
            original_pointer = pointer.read_bytes()
            pointer.write_text(str(linked_parent / content.name) + "\n")
            try:
                code, refused = _run(target, *args, "--plan")
                assert code == 1 and "link" in refused["error"], refused
            finally:
                pointer.write_bytes(original_pointer)
        assert not target.exists() and not store.exists()
    finally:
        if sys.platform == "win32":
            linked_parent.rmdir()
        else:
            linked_parent.unlink()
    if not in_process:
        distribution = sites[0] / "removed-9.0.dist-info"
        saved = tmp_path / "saved-distribution"
        distribution.rename(saved)
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(distribution), str(saved)],
                capture_output=True,
                check=True,
            )
        else:
            distribution.symlink_to(saved, target_is_directory=True)
        code, refused = _run(target, *args, "--plan")
        assert code == 1 and "linked dependency metadata" in refused["error"], refused
        if sys.platform == "win32":
            distribution.rmdir()
        else:
            distribution.unlink()
        saved.rename(distribution)
        assert not target.exists() and not store.exists()
    for destination in (source, source / "child", tmp_path):
        code, refused = _run(destination, *args, "--yes")
        assert code == 1, refused
    code, refused = _run(target, *args, "--in-process-pack", "different", "--yes")
    assert code == 1 and "preserve" in refused["error"], refused
    code, refused = _run(target, *args, "--data", str(source / "new-store"), "--yes")
    assert code == 1 and "disjoint" in refused["error"], refused
    retained.write_bytes(b"changed archive")
    code, corrupt = _run(target, *args, "--yes")
    assert code == 1 and "digest differs" in corrupt["error"], corrupt
    assert not target.exists() and not store.exists()
    if not in_process:
        shutil.copyfile(archive, retained)

        def confirmed(*_args):
            metadata = sites[0] / "removed-9.0.dist-info/METADATA"
            metadata.write_text("Name: removed\nVersion: 8.0\n\n")
            return True

        monkeypatch.setattr(environment, "_confirmed", confirmed)
        monkeypatch.setattr(sys, "argv", ["dinkster-env", "--root", str(target), *args, "--json"])
        with pytest.raises(SystemExit) as stopped:
            environment.main()
        assert stopped.value.code == 1
        refused = json.loads(capsys.readouterr().out)
        assert "standalone inputs changed" in refused["error"]
        assert not (target / "current").exists() and history(target) == []
        assert not (store / "bases").exists()


def test_adoption_refuses_source_changed_after_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from dinkster_registry import Lockfile
    from dinkster_supervisor import environment
    from test_environment_artifacts import _mirror

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    source, target, store, mirror = (tmp_path / name for name in ("old", "new", "store", "mirror"))
    (source / "generations").mkdir(parents=True)
    (source / "generations/1.json").write_text(Lockfile().record_json())
    (source / "generations/2.json").write_text(Lockfile().record_json())
    (source / "current").write_text("1\n")
    base, _, channel = _mirror(mirror)

    def confirmed(*_args):
        (source / "current").write_text("2\n")
        return True

    monkeypatch.setattr(environment, "_confirmed", confirmed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-env",
            "--root",
            str(target),
            "setup",
            "--adopt-from",
            str(source),
            "--variant",
            "fixture-cpu",
            "--channel",
            "main",
            "--mirror",
            str(mirror),
            "--data",
            str(store),
            "--engine-release",
            channel["engine"]["release"],
            "--base-id",
            base["id"],
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        environment.main()
    assert stopped.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert "standalone inputs changed" in result["error"]
    assert not (target / "current").exists()
    assert history(target) == []
    assert not (store / "bases").exists()


@pytest.mark.parametrize("in_process", [False, True], ids=["isolated", "shared-in-process"])
def test_restore_plan_reports_only_required_pack_wheels_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
):
    from dinkster_supervisor.environment_state import content_digest
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "fixture"),
            ],
            text=True,
            timeout=120,
        )
    )
    root, store = Path(fixture["root"]), Path(fixture["store"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, (
        result,
        {
            str(path.relative_to(root)): path.read_text(encoding="utf-8", errors="replace")
            for path in (root / "runtime").glob("*/*.log")
        },
    )
    assert _run(root, "stop", "--yes")[0] == 0
    source = wheel(tmp_path, "required_extra", "1.0", 73)
    payload = source.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    required = LockedWheel(source.name, len(payload), sha, source.as_uri())
    incompatible = replace(
        required,
        filename="required_extra-1.0-cp999-cp999-any.whl",
        url=(tmp_path / "required_extra-1.0-cp999-cp999-any.whl").as_uri(),
    )
    generation = root / "generations/1"
    path = generation / "packs/example/pack-lock.json"
    path.parent.mkdir(parents=True)
    PackLock(
        "example",
        Generation.read(generation / "generation.json").variant,
        ">=3.12",
        (),
        (),
        (
            LockedDistribution("fixture-base", "1.0", ()),
            LockedDistribution("required-extra", "1.0", (incompatible, required)),
        ),
    ).write(path)
    record = Generation.read(generation / "generation.json")
    files = [*record.files, ("packs/example/pack-lock.json", content_digest(path))]
    if in_process:
        shared = generation / "packs/zzz/pack-lock.json"
        shared.parent.mkdir()
        replace(PackLock.read(path), pack="zzz").write(shared)
        files.append(("packs/zzz/pack-lock.json", content_digest(shared)))
    replace(record, files=tuple(files), in_process_packs=("zzz",) if in_process else ()).write(
        generation / "generation.json"
    )
    entry = prepare_entry(root, generation, operation="pack-fixture", reason="pack-install")
    publish_entry(root, entry)
    assert "py3-none-any" in cast("dict", entry["snapshot"])["host"]["wheelTags"]
    preserved = (root / "history/00000000000000000002.json").read_bytes()
    source.unlink()
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".lock" and "runtime" not in path.relative_to(root).parts
    }
    code, unavailable = _run(root, "restore", "2", "--yes")
    assert code == 0 and unavailable["status"] == "approval-required", unavailable
    assert unavailable["restorePlan"]["missing"] == [
        {
            "kind": "pack dependency",
            "identity": "zzz" if in_process else "example",
            "artifact": source.name,
            "sha256": sha,
            "locations": [source.as_uri()],
        }
    ]
    assert {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".lock" and "runtime" not in path.relative_to(root).parts
    } == before
    cached = object_path(store / "wheels", "sha256:" + sha) / source.name
    cached.parent.mkdir()
    for content, exact in ((b"x" * len(payload), False), (payload, True)):
        cached.write_bytes(content)
        code, planned = _run(root, "restore", "2", "--plan")
        assert code == 0, planned
        assert planned["restorePlan"]["exact"] is exact
        assert (root / "current").read_text() == "1\n"
        assert (root / "history/00000000000000000002.json").read_bytes() == preserved

    journal = next((root / "journal").glob("*.json"))
    replace(Journal.read(journal), history_entry=None).write(journal)
    for path in (root / "history").glob("*.json"):
        path.unlink()
    code, result = _run(root, "import", "1", "--yes")
    assert code == 0, result
    imported = result["entry"]
    assert imported["snapshot"]["host"] == {
        "system": "unknown",
        "machine": "unknown",
        "python": "unknown",
    }
    assert "wheelTags" not in imported["observedWith"]["host"]
    original = (root / "history/00000000000000000001.json").read_bytes()
    code, result = _run(root, "restore", "1", "--yes")
    assert code == 1 and "wheel compatibility tags" in result["error"], result
    assert (root / "current").read_text() == "1\n"
    assert (root / "history/00000000000000000001.json").read_bytes() == original


@pytest.mark.parametrize("combined", [False, True], ids=["base-only", "engine-and-base"])
def test_missing_base_offers_minimal_compatible_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, combined: bool
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "fixture"),
            ],
            text=True,
            timeout=120,
        )
    )
    root, store, mirror = Path(fixture["root"]), Path(fixture["store"]), Path(fixture["mirror"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        first = history(root)[0]
        snapshot = cast("dict", first["snapshot"])
        base_record = snapshot["base"]
        original = json.loads(snapshot["metadata"]["base-manifest.json"])
        published = mirror / "base" / base_record["variant"] / base_record["id"][7:]
        index = [{"id": "sha256:" + "f" * 64, "path": f"base/{base_record['variant']}/missing"}]
        replacement_id = None
        for compatible in (False, True):
            base = json.loads(json.dumps(original))
            base["interpreter"]["release"] = "replacement" if compatible else "incompatible"
            if not compatible:
                base["python"] = "3.99.1"
            identity_fields = {
                key: value for key, value in base.items() if key not in {"id", "signature"}
            }
            identity_fields["archive"] = {"format": "tar+zstd"}
            identity = hashlib.sha256(canonical_json(identity_fields)).hexdigest()
            base["id"] = "sha256:" + identity
            destination = published.parent / identity
            shutil.copytree(published, destination)
            (destination / "manifest.json").write_bytes(canonical_json(base))
            index.append({"id": base["id"], "path": f"base/{base_record['variant']}/{identity}"})
            if compatible:
                replacement_id = base["id"]
        (published.parent / "index.json").write_text(json.dumps(index))
        shutil.copyfile(fixture["updatedChannel"], mirror / "channels/main.json")
        code, result = _run(root, "update", "--yes")
        assert code == 0 and result["current"] == 2, result
        expected_engine = snapshot["engine"]
        if combined:
            scratch = tmp_path / "replacement-base"
            scratch.mkdir()
            base = build_base(mirror, scratch, base_record["variant"], version="1.1")
            replacement_id = base["id"]
            index.append(
                {
                    "id": replacement_id,
                    "path": f"base/{base_record['variant']}/{replacement_id[7:]}",
                }
            )
            (published.parent / "index.json").write_text(json.dumps(index))
            executable_release(mirror, base, "3.0", 73)
            channel_path = mirror / "channels/main.json"
            channel = json.loads(channel_path.read_bytes())
            expected_engine = {key: channel["engine"][key] for key in ("version", "release")}
            channel["base"][base_record["variant"]] = {
                "id": replacement_id,
                "locations": [
                    {
                        "type": "mirror",
                        "path": f"base/{base_record['variant']}/{replacement_id[7:]}/manifest.json",
                    }
                ],
            }
            channel_path.write_bytes(canonical_json(channel))
            # Neither replacement works with its historical counterpart.
            for item in index[1:-1]:
                shutil.rmtree(mirror / item["path"])
            old_release = snapshot["engine"]["release"][7:]
            shutil.rmtree(mirror / "engine" / old_release)
            manifest = json.loads(snapshot["metadata"]["engine-manifest.json"])
            for row in manifest["artifacts"]:
                if row["name"].endswith(".whl"):
                    shutil.rmtree(object_path(store / "wheels", "sha256:" + row["sha256"]))
        preserved = {path.name: path.read_bytes() for path in (root / "history").glob("*.json")}
        shutil.rmtree(published)
        shutil.rmtree(object_path(store / "bases", base_record["id"]))
        shutil.rmtree(object_path(store / "downloads", base_record["id"]))
        before = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != ".lock"
            and "runtime" not in path.relative_to(root).parts
        }
        code, result = _run(root, "restore", "1", "--yes")
        assert code == 0 and result["status"] == "approval-required", result
        alternatives = result["restorePlan"]["alternatives"]
        assert len(alternatives) == 1
        chosen = alternatives[0]
        assert chosen["snapshot"]["engine"] == expected_engine
        assert chosen["snapshot"]["base"]["id"] == replacement_id
        expected_wheels = []
        base_manifest = json.loads(chosen["snapshot"]["metadata"]["base-manifest.json"])
        expected_wheels.extend(
            {key: row[key] for key in ("filename", "size", "sha256")}
            for row in base_manifest["wheels"]
        )
        release = mirror / "engine" / expected_engine["release"][7:]
        for path in release.glob("*.whl"):
            expected_wheels.append(
                {
                    "filename": path.name,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        expected_receipt = {
            "format": "dinkster.installed-wheels/1",
            "wheels": sorted(expected_wheels, key=lambda row: row["filename"]),
            "inherited": [],
        }
        assert (
            json.loads(chosen["snapshot"]["metadata"]["installed-wheels.json"]) == expected_receipt
        )
        base_change = {
            "field": "base",
            "before": base_record,
            "after": {"id": replacement_id, "variant": base_record["variant"]},
        }
        assert base_change in chosen["substitutions"]
        assert {row["field"] for row in chosen["substitutions"]} == (
            {"engine", "base", "constraints"} if combined else {"base"}
        )
        if combined:
            assert next(row for row in chosen["substitutions"] if row["field"] == "engine") == {
                "field": "engine",
                "before": snapshot["engine"],
                "after": expected_engine,
            }
        else:
            assert chosen["substitutions"] == [base_change]
        assert {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.name != ".lock"
            and "runtime" not in path.relative_to(root).parts
        } == before
        code, result = _run(root, "restore", "1", "--approve", chosen["approval"])
        assert code == 0 and result["current"] == 3 and result["exact"] is False, result
        restored = history(root)[-1]
        assert cast("dict", restored["snapshot"])["engine"] == expected_engine
        assert cast("dict", restored["snapshot"])["base"]["id"] == replacement_id
        assert (
            json.loads(cast("dict", restored["snapshot"])["metadata"]["installed-wheels.json"])
            == expected_receipt
        )
        assert restored["substitutions"] == chosen["substitutions"]
        assert {
            path.name: path.read_bytes()
            for path in (root / "history").glob("*.json")
            if path.name in preserved
        } == preserved
    finally:
        _run(root, "stop", "--yes")


@pytest.mark.parametrize("readable", [False, True], ids=["legacy", "readable"])
def test_local_restore_reuses_retained_inputs_without_checkout_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readable: bool
):
    from dinkster_supervisor.environment_generations import build_generation
    from dinkster_supervisor.environment_state import content_digest, write_document
    from dinkster_supervisor.environment_transactions import activate, begin_activation

    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    fixture = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "fixture"),
            ],
            text=True,
            timeout=120,
        )
    )
    root, store = Path(fixture["root"]), Path(fixture["store"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        first = Generation.read(root / "generations/1/generation.json")
        old_base = object_path(store / "bases", first.base)
        staging = tmp_path / "local-base"
        shutil.copytree(old_base / "python", staging / "python", symlinks=False)
        base = json.loads((old_base / "manifest.json").read_bytes())
        base.pop("id")
        base.pop("archive")
        base.update(source="local")
        base["interpreter"]["files"] = {
            path.relative_to(staging).as_posix(): content_digest(path)
            for path in (staging / "python").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        base_id = "sha256:" + hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
        base["id"] = base_id
        write_document(staging / "manifest.json", base)
        shutil.move(
            str(staging),
            object_path(store / "bases", base_id, label="local-base" if readable else None),
        )
        published = json.loads((root / "generations/1/engine-manifest.json").read_bytes())
        engine = {
            "format": "dinkster.engine/1",
            "source": "local",
            "version": first.engine_version,
            "workerProtocol": first.worker_protocol,
            "wheels": [
                {**row, "filename": row["name"], "url": "local"}
                for row in published["artifacts"]
                if row["name"].endswith(".whl")
            ],
        }
        engine_id = (
            "sha256:" + hashlib.sha256(json.dumps(engine, sort_keys=True).encode()).hexdigest()
        )
        write_document(
            object_path(store / "engines", engine_id, label="local-engine" if readable else None)
            / "release-manifest.json",
            engine,
        )
        journal = begin_activation(root, 2, "update")
        build_generation(
            root=root,
            number=2,
            store=store,
            base=base_id,
            engine_version=first.engine_version,
            engine_release=engine_id,
            constraints=root / "generations/1/constraints.txt",
            variant=first.variant,
            worker_protocol=first.worker_protocol,
            source_kind="local",
        )
        activate(root, journal, revalidate=lambda: None)
        before = history(root)
        assert cast("dict[str, object]", before[-1]["snapshot"])["source"] == "local"
        code, result = _run(root, "restore", "1", "--yes")
        assert code == 0 and result["current"] == 3, result
        shutil.rmtree(root / "generations/2")
        code, planned = _run(root, "restore", "2", "--plan")
        assert code == 0 and planned["restorePlan"]["exact"] is True, planned
        code, restored = _run(root, "restore", "2", "--yes")
        assert code == 0 and restored["current"] == 4 and restored["exact"] is True, restored
        assert history(root)[-1]["snapshot"] == before[-1]["snapshot"]
        assert history(root)[:2] == before
        for path, kind in (
            (
                object_path(store / "bases", base_id) / next(iter(base["interpreter"]["files"])),
                "local base",
            ),
            (
                object_path(store / "wheels", "sha256:" + engine["wheels"][0]["sha256"])
                / engine["wheels"][0]["filename"],
                "local wheel",
            ),
        ):
            payload = path.read_bytes()
            path.unlink()
            code, unavailable = _run(root, "restore", "2", "--yes")
            assert code == 0 and unavailable["status"] == "approval-required", unavailable
            assert any(item["kind"] == kind for item in unavailable["restorePlan"]["missing"])
            assert (root / "current").read_text() == "4\n"
            assert len(history(root)) == 4
            path.write_bytes(payload)
    finally:
        _run(root, "stop", "--yes")


def test_history_preserves_software_and_diffs_after_generation_is_pruned(tmp_path: Path):
    first_generation = _generation(tmp_path, 1).parent
    for name in ("workflows", "models", "outputs"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "private").write_bytes(b"user data must not enter the software snapshot")
    first = prepare_entry(tmp_path, first_generation, operation="install", reason="setup")
    publish_entry(tmp_path, first)
    initial_bytes = (tmp_path / "history/00000000000000000001.json").read_bytes()
    second_generation = _generation(tmp_path, 2).parent
    record = Generation.read(second_generation / "generation.json")
    replace(record, engine_version="0.4.2", engine_release="sha256:" + "d" * 64).write(
        second_generation / "generation.json"
    )
    second = prepare_entry(tmp_path, second_generation, operation="update", reason="update")
    publish_entry(tmp_path, second)
    shutil.rmtree(first_generation)
    code, result = _run(tmp_path, "history")
    assert code == 0
    assert result["history"] == [first, second]
    assert second["diff"] == [
        {"field": "engine.release", "before": "sha256:" + "b" * 64, "after": "sha256:" + "d" * 64},
        {"field": "engine.version", "before": "0.3.1", "after": "0.4.2"},
    ]
    code, result = _run(tmp_path, "history", "show", "1")
    assert code == 0 and result["entry"] == first
    snapshot = cast("dict[str, object]", first["snapshot"])
    assert snapshot["metadata"] == {"constraints.txt": "demo==1\n"}
    assert b"user data must not enter" not in initial_bytes
    for name in ("workflows", "models", "outputs"):
        assert (tmp_path / name / "private").read_bytes() == (
            b"user data must not enter the software snapshot"
        )
    publish_entry(tmp_path, first)
    assert (tmp_path / "history/00000000000000000001.json").read_bytes() == initial_bytes
    with pytest.raises(EnvironmentError, match="cannot replace"):
        publish_entry(tmp_path, {**first, "reason": "rewritten"})
    assert history(tmp_path) == [first, second]
    assert _run(tmp_path, "history", "show", "99")[0] == 1


def test_snapshot_refuses_modified_metadata_and_linked_pack_parents(tmp_path: Path):
    generation = _generation(tmp_path, 1).parent
    (generation / "constraints.txt").write_bytes(b"demo==2\n")
    with pytest.raises(EnvironmentError, match="intact software metadata: constraints.txt"):
        prepare_entry(tmp_path, generation, operation="install", reason="setup")
    assert history(tmp_path) == []
    assert not (tmp_path / "history").exists()
    (generation / "constraints.txt").write_bytes(b"demo==1\n")
    target = tmp_path / "user-files"
    target.mkdir()
    (target / "private").write_text("not software")
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(generation / "packs"), str(target)],
            capture_output=True,
            check=True,
        )
    else:
        (generation / "packs").symlink_to(target, target_is_directory=True)
    document = json.loads((generation / "generation.json").read_bytes())
    document["files"]["packs/private/pack-lock.json"] = "sha256:" + "e" * 64
    (generation / "generation.json").write_text(json.dumps(document))
    with pytest.raises(EnvironmentError, match="intact software metadata: packs/private"):
        prepare_entry(tmp_path, generation, operation="install", reason="setup")
    assert (target / "private").read_text() == "not software"


@pytest.mark.parametrize(
    "operation,interrupted_update", [("restore", False), ("rollback", False), ("rollback", True)]
)
def test_restore_rebuilds_pruned_software_without_changing_user_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, interrupted_update: bool
):
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    monkeypatch.setenv("DINKSTER_TEST_FORBID_RESOLUTION", "1")
    fixture = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).parent / "fixtures/environments/build.py"),
                "--output",
                str(tmp_path / "fixture"),
            ],
            text=True,
            timeout=120,
        )
    )
    root = Path(fixture["root"])
    setup = fixture["commands"]["setup"]
    code, result = _run(root, *setup[setup.index("setup") : -1])
    assert code == 0, result
    try:
        first = history(root)[0]
        preserved = (root / "history/00000000000000000001.json").read_bytes()
        code, preference = _run(root, "cleanup", "--automatic", "off", "--yes")
        assert code == 0 and preference["cleanup"]["policy"]["automatic"] is False, preference
        cleanup_policy = (root / "cleanup.json").read_bytes()
        for name in ("workflows", "models", "outputs"):
            directory = root / name
            directory.mkdir(exist_ok=True)
            (directory / "sentinel").write_bytes(name.encode("ascii") + b" private data")
        shutil.copyfile(fixture["updatedChannel"], Path(fixture["mirror"]) / "channels/main.json")
        if interrupted_update:
            script = """
import os, sys
from dinkster_supervisor import environment, environment_transactions
original = environment_transactions.publish_entry
def interrupt(root, entry):
    if entry['id'] == 2:
        os._exit(74)
    original(root, entry)
environment_transactions.publish_entry = interrupt
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'update', '--yes', '--json']
environment.main()
"""
            completed = subprocess.run(
                [sys.executable, "-c", script, str(root)], capture_output=True, timeout=120
            )
            assert completed.returncode == 74, completed.stderr
            assert (root / "current").read_text() == "2\n"
            assert len(history(root)) == 1
            code, result = _run(root, "rollback", "--plan")
            assert code == 0 and result["status"] == "planned", result
            assert len(history(root)) == 2
            assert (root / "current").read_text() == "2\n"
        else:
            code, result = _run(root, "update", "--yes")
            assert code == 0 and result["current"] == 2, result
        second = (root / "history/00000000000000000002.json").read_bytes()
        assert generation_history_coverage(root, 1) == 1
        assert generation_history_coverage(root, 2) == 2
        parked_port = free_port()
        assert (
            _run(root, "start", "--engine", "parked", "--port", str(parked_port), "--yes")[0] == 0
        )
        assert _run(root, "stop", "--engine", "parked", "--yes")[0] == 0
        shutil.rmtree(root / "generations/1")
        project = json.loads((root / "project.json").read_bytes())
        project.update(channel="stable", custom={"preserve": "edited after engine update"})
        (root / "project.json").write_text(json.dumps(project))
        settings = (root / "project.json").read_bytes()
        command = ("restore", "1") if operation == "restore" else ("rollback",)
        code, result = _run(root, *command, "--plan")
        assert code == 0 and result["status"] == "planned", result
        assert len(history(root)) == 2 and not (root / "generations/3").exists()
        code, result = _run(root, *command, "--yes")
        assert code == 0 and result["current"] == 3 and result["exact"] is True, result
        entries = history(root)
        assert len(entries) == 3 and entries[-1]["restoredFrom"] == 1
        assert entries[-1]["reason"] == operation
        assert entries[-1]["snapshot"] == first["snapshot"]
        assert entries[-1]["recordedTools"] == cast("dict", first["snapshot"])["tools"]
        assert generation_history_coverage(root, 3) == 3
        assert engine_status(root)[1] == {
            "name": "parked",
            "generation": 3,
            "state": "stopped",
            "url": f"http://127.0.0.1:{parked_port}",
        }
        assert (root / "history/00000000000000000001.json").read_bytes() == preserved
        assert (root / "history/00000000000000000002.json").read_bytes() == second
        assert (root / "project.json").read_bytes() == settings
        assert (root / "cleanup.json").read_bytes() == cleanup_policy
        code, status = _run(root, "status")
        assert code == 0 and status["cleanup"]["policy"]["automatic"] is False, status
        monkeypatch.setenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM", "refuse")
        code, failed = _run(root, *command, "--yes")
        assert code == 1 and "recorded by tools" in failed["error"], failed
        assert (root / "current").read_text() == "3\n"
        assert history(root) == entries and not (root / "generations/4").exists()
        code, queried = _run(root, "history")
        assert code == 0 and len(queried["attempts"]) == 1
        failed_source = 1 if operation == "restore" else 2
        assert queried["attempts"][0]["restoredFrom"] == failed_source
        assert queried["attempts"][0]["error"] == failed["error"]
        monkeypatch.delenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
        interrupt_restore = """
import os
import sys
from dinkster_supervisor import environment, environment_restore
built = sys.argv[2] == 'built'
original = environment_restore.build_generation
def interrupt(**kwargs):
    if built:
        original(**kwargs)
    os._exit(73)
environment_restore.build_generation = interrupt
command = ['restore', '1'] if sys.argv[3] == 'restore' else ['rollback']
sys.argv = ['dinkster-env', '--root', sys.argv[1], *command, '--yes', '--json']
environment.main()
"""
        for count, stage in enumerate(("building", "built"), 2):
            interrupted = subprocess.run(
                [sys.executable, "-c", interrupt_restore, str(root), stage, operation], timeout=120
            )
            assert interrupted.returncode == 73
            assert (root / "current").read_text() == "3\n" and history(root) == entries
            if stage == "building":
                interrupted_recovery = """
import os
import sys
from dinkster_supervisor import environment, environment_transactions
original = environment_transactions.record_failed_restore
def interrupt(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(74)
environment_transactions.record_failed_restore = interrupt
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'stop', '--yes', '--json']
environment.main()
"""
                recovered_once = subprocess.run(
                    [sys.executable, "-c", interrupted_recovery, str(root)], timeout=120
                )
                assert recovered_once.returncode == 74
                assert len(_run(root, "history")[1]["attempts"]) == count
            assert _run(root, "stop", "--yes")[0] == 0
            assert not (root / "generations/4").exists()
            code, recovered = _run(root, "history")
            assert code == 0 and len(recovered["attempts"]) == count
            assert recovered["attempts"][-1]["restoredFrom"] == failed_source
            assert "interrupted" in recovered["attempts"][-1]["error"]
            assert history(root) == entries
            assert _run(root, "stop", "--yes")[0] == 0
            assert _run(root, "history")[1]["attempts"] == recovered["attempts"]
        old_snapshot = cast("dict[str, object]", first["snapshot"])
        old_engine = cast("dict[str, str]", old_snapshot["engine"])
        old_metadata = cast("dict[str, str]", old_snapshot["metadata"])
        old_manifest = json.loads(old_metadata["engine-manifest.json"])
        store = Path(fixture["store"])
        mirror = Path(fixture["mirror"])
        executable_release(
            mirror, json.loads(Path(fixture["baseManifest"]).read_bytes()), "3.0", 73
        )
        shutil.copyfile(fixture["updatedChannel"], mirror / "channels/main.json")
        shutil.rmtree(mirror / "engine" / old_engine["release"][7:])
        (mirror / "engine/index.json").write_text(
            json.dumps(
                [
                    {
                        "path": f"engine/{old_engine['release'][7:]}",
                        "sha256": old_engine["release"][7:],
                    },
                    *[
                        {"path": f"engine/{path.parent.name}", "sha256": path.parent.name}
                        for path in sorted((mirror / "engine").glob("*/release-manifest.json"))
                    ],
                ]
            )
        )
        for artifact in old_manifest["artifacts"]:
            if artifact["name"].endswith(".whl"):
                cached = object_path(store / "wheels", "sha256:" + artifact["sha256"])
                if cached.exists():
                    shutil.rmtree(cached)
        prior_files = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.parts[-1] != ".lock"
            and "runtime" not in path.relative_to(root).parts
        }
        for flags in (("--plan",), (), ("--yes",)):
            code, proposed = _run(root, "restore", "1", *flags)
            assert code == 0 and proposed["status"] == "approval-required", proposed
            assert any(
                item["identity"] == old_engine["release"]
                for item in proposed["restorePlan"]["missing"]
            )
            assert {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
                and path.parts[-1] != ".lock"
                and "runtime" not in path.relative_to(root).parts
            } == prior_files
        alternatives = proposed["restorePlan"]["alternatives"]
        assert len(alternatives) == 2
        assert {item["snapshot"]["engine"]["version"] for item in alternatives} == {"2.0", "3.0"}
        for item in alternatives:
            assert {change["field"] for change in item["substitutions"]} == {
                "engine",
                "constraints",
            }
        chosen = next(
            item for item in alternatives if item["snapshot"]["engine"]["version"] == "2.0"
        )
        approval = chosen["approval"]
        (root / "project.json").write_bytes(settings + b"\n")
        code, stale = _run(root, "restore", "1", "--approve", approval)
        assert code == 1 and "stale" in stale["error"], stale
        assert (root / "current").read_text() == "3\n"
        (root / "project.json").write_bytes(settings)
        code, restored = _run(root, "restore", "1", "--approve", approval)
        assert code == 0 and restored["current"] == 4 and restored["exact"] is False, restored
        assert len(history(root)) == 4
        approximate = history(root)[-1]
        assert approximate["exact"] is False and approximate["restoredFrom"] == 1
        assert approximate["substitutions"] == chosen["substitutions"]
        assert cast("dict[str, object]", approximate["snapshot"])["base"] == old_snapshot["base"]
        for name in ("workflows", "models", "outputs"):
            assert (root / name / "sentinel").read_bytes() == name.encode(
                "ascii"
            ) + b" private data"
        assert _run(root, "doctor")[1]["status"] == "clean"
    finally:
        _run(root, "stop", "--yes")
