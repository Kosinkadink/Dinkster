from __future__ import annotations

import builtins
import contextlib
import json
import time
from pathlib import Path
from typing import cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import (
    MountConfiguration,
    MountDef,
    MountTable,
    dump_mounts,
    load_mounts,
    load_output_mount,
)
from dinkster_library import LibraryClient

from dinkster import frontend, launch, setup
from dinkster.cli import main as cli_main
from dinkster.frontend import install_frontend
from dinkster.mounts_api import MountService
from dinkster.project_folders import ProjectFolders
from tests.async_scenarios import run_scenario
from tests.fixtures.environments.build import channel_json, read_channel
from tests.supervisor_subprocess import run as run_with_diagnostics

pytestmark = pytest.mark.usefixtures("signed_channels")


def test_library_initialization_creates_default_mounts(tmp_path: Path) -> None:
    library = tmp_path / "library"
    setup.ensure_default_mounts(library)
    assert library.is_dir()
    assert (library / "output").is_dir()
    assert (library / "imports").is_dir()
    assert load_mounts(library / "mounts.toml") == (
        MountDef(id="imports", path=library / "imports", mode="readwrite"),
        MountDef(id="output", path=library / "output", mode="readwrite"),
    )
    assert load_output_mount(library / "mounts.toml") == "output"
    table = MountTable(output_mount="output")
    for mount in load_mounts(library / "mounts.toml"):
        table.add(mount)
    table.add(MountDef("derived", tmp_path / "derived"), source="derived")
    service = MountService(table, library / "mounts.toml")
    assert set(service.project_context("default").mounts) == {"derived", "imports", "output"}


@pytest.mark.parametrize(
    "mode",
    [
        "json",
        "text",
        "newer-inference",
        "newer-variant-inference",
        "missing-locks",
        "no-defaults",
        "offline",
        "offline-metadata",
        "offline-registry",
        "corrupt-cache",
        "decline",
        "nonterminal",
        "changed-lock",
    ],
)
def test_setup_plan_uses_target_defaults_and_bound_locks_without_acquiring_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, mode: str
) -> None:
    import hashlib

    from dinkster_registry import LockedPack, artifact_digest
    from dinkster_store import object_path
    from dinkster_supervisor.environment_artifacts import acquire, discover
    from dinkster_workers.pack_lock import PackLock
    from test_environment_artifacts import _artifact, _json, _mirror
    from test_installer import _serve_artifacts, _static_indexes, _static_release

    from dinkster.registries import (
        NamedRegistry,
        RegistryConfig,
        StaticPackIndex,
        plan_registry_sources,
    )

    mirror, store, root = tmp_path / "mirror", tmp_path / "store", tmp_path / "project"
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA", "DINKSTER_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setenv("DINKSTER_HOME", str(root.parent))
    monkeypatch.delenv("DINKSTER_REGISTRY", raising=False)
    monkeypatch.delenv("DINKSTER_REGISTRIES", raising=False)
    base, engine, channel = _mirror(mirror)
    defaults = {"example-pack": "1.2.0", "other-pack": "1.2.3"}
    if mode in {"newer-inference", "newer-variant-inference"}:
        engine["artifacts"].append(
            _artifact("dinkster_inference-0.1.2-py3-none-any.whl", b"inference")
        )
    if mode != "no-defaults":
        engine["defaultPacks"] = defaults
    previous = mirror / "engine" / channel["engine"]["release"][7:]
    payload = _json(engine)
    release = hashlib.sha256(payload).hexdigest()
    previous.rename(previous.with_name(release))
    (previous.with_name(release) / "release-manifest.json").write_bytes(payload)
    channel["engine"]["release"] = "sha256:" + release
    channel["engine"]["locations"] = [
        {"type": "mirror", "path": f"engine/{release}/release-manifest.json"}
    ]
    (mirror / "channels/nightly.json").write_bytes(channel_json(channel))
    files: dict[str, bytes] = {}
    releases = []
    for name in defaults:
        entry = LockedPack(name, "1.2.3", artifact_digest(name.encode()), "acme", ())
        row = _static_release(entry, name.encode())
        lock = PackLock(name, "fixture-cpu", ">=3.12", (), (), ())
        lock_bytes = lock.to_json().encode()
        identity = artifact_digest(lock_bytes)
        path = "locks/" + identity.replace(":", "/") + ".json"
        row["installLocks"] = (
            {}
            if mode == "missing-locks"
            else {
                "fixture-cpu": {
                    "baseId": base["id"],
                    "artifactDigest": entry.artifact_digest,
                    "digest": identity,
                    "size": len(lock_bytes),
                    "url": path,
                }
            }
        )
        files["/static/" + path] = lock_bytes
        releases.append(row)
        if mode in {"newer-inference", "newer-variant-inference"} and name == "example-pack":
            newer = _static_release(
                LockedPack(name, "2.0", artifact_digest(b"newer pack"), "acme", ()), b"newer pack"
            )
            if mode == "newer-inference":
                newer["requires"] = ["dinkster-inference>=0.1.3"]
            else:
                variant_lock = (
                    PackLock(name, "fixture-cpu", ">=3.12", ("dinkster-inference>=0.1.3",), (), ())
                    .to_json()
                    .encode()
                )
                identity = artifact_digest(variant_lock)
                path = "locks/" + identity.replace(":", "/") + ".json"
                newer["installLocks"] = {
                    "fixture-cpu": {
                        "baseId": base["id"],
                        "artifactDigest": artifact_digest(b"newer pack"),
                        "digest": identity,
                        "size": len(variant_lock),
                        "url": path,
                    }
                }
                files["/static/" + path] = variant_lock
            releases.append(newer)
    files.update(_static_indexes(releases))
    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(files, requests)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    registry_file = tmp_path / "registries.toml"
    registry_file.write_text(f'[registries.private]\nendpoint = "{endpoint}"\ndefault = true\n')
    try:
        if mode.startswith("offline"):
            acquire(discover((str(mirror),), "nightly"), "fixture-cpu", store)
            registry = NamedRegistry("private", endpoint, default=True)
            plan_registry_sources(
                RegistryConfig((registry,)),
                registry,
                defaults,
                engine_version=engine["version"],
                variant="fixture-cpu",
                base_id=base["id"],
                index=StaticPackIndex(store / "metadata" / "registry"),
            )
            if mode == "offline-metadata":
                for identity in (base["id"], channel["engine"]["release"]):
                    (store / "metadata" / f"{identity[7:]}.json").unlink()
            elif mode == "offline-registry":
                for metadata in (store / "metadata/registry").rglob("*"):
                    if metadata.is_file():
                        metadata.unlink()
            files.clear()
            mirror.rename(tmp_path / "disconnected")
            requests.clear()
        cached = None
        if mode in {"offline", "corrupt-cache"}:
            cached = (
                object_path(
                    store / "wheels", "sha256:" + hashlib.sha256(b"example-pack").hexdigest()
                )
                / "example_pack-1.2.3-py3-none-any.whl"
            )
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"example-pack" if mode == "offline" else b"X" * 12)
        before = {p: p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")}
        arguments = [
            "setup",
            "--plan",
            "--data",
            str(store),
            "--mirror",
            str(mirror),
            "--channel",
            "nightly",
            "--variant",
            "fixture-cpu",
            "--registries",
            str(registry_file),
        ]
        if mode in {"decline", "nonterminal", "changed-lock"}:
            arguments.remove("--plan")
            monkeypatch.setattr("sys.stdin.isatty", lambda: mode != "nonterminal")

            def answer(_prompt: str) -> str:
                if mode == "changed-lock":
                    for path in list(files):
                        if "/locks/" in path:
                            files[path] = b"changed lock metadata"
                return "n" if mode == "decline" else "y"

            monkeypatch.setattr(builtins, "input", answer)
        if mode != "text":
            arguments.append("--json")
        if mode.startswith("offline"):
            arguments.append("--offline")
        code = cli_main(arguments)
        captured = capsys.readouterr()
        if mode == "offline-metadata":
            assert code == 1
            assert base["id"] in captured.err
            assert channel["engine"]["release"] in captured.err
        elif mode == "offline-registry":
            assert code == 1
            assert "example-pack" in captured.err and "other-pack" in captured.err
        elif mode == "no-defaults":
            assert code == 1
            assert "target engine has no default pack declaration" in captured.err
            assert requests == []
        elif mode == "missing-locks":
            assert code == 1
            assert "example-pack@1.2.3" in captured.err and "other-pack@1.2.3" in captured.err
        elif mode in {"decline", "nonterminal", "changed-lock"}:
            assert code == (0 if mode == "decline" else 1)
            assert "Engine binding:" in captured.err
            assert "Dependency lock:" in captured.err
            if mode == "nonterminal":
                assert "--yes" in captured.err and "requires a terminal" in captured.err
            if mode == "changed-lock":
                assert "refused" in captured.err
        else:
            assert code == 0, captured.err
            if mode == "text":
                assert f"Engine: {engine['version']} (sha256:{release})" in captured.out
                assert "Runtime: Python 3.14.10; fixture-cpu" in captured.out
                assert f"Pack: example-pack@1.2.3; registry:{endpoint}" in captured.out
                assert "Dependency lock: other-pack; sha256:" in captured.out
                assert "Pack payload upper bound: 22 bytes" in captured.out
            else:
                result = json.loads(captured.out)
                assert result["engineRelease"] == "sha256:" + release
                assert result["baseId"] == base["id"]
                assert set(result["dependencyLocks"]) == set(defaults)
                assert result["maximumPackPayloadBytes"] == 22
                if mode in {"newer-inference", "newer-variant-inference"}:
                    assert result["sources"]["packs"][0]["version"] == "1.2.3"
                    assert "example-pack@2.0 skipped" in result["skippedPackVersions"][0]
                    assert "dinkster-inference>=0.1.3" in result["skippedPackVersions"][0]
                    assert "dinkster-inference==0.1.2" in result["skippedPackVersions"][0]
                    assert result["skippedPackVersions"][0] in result["plan"]
                assert all(
                    row["source"] == f"registry:{endpoint}" for row in result["sources"]["packs"]
                )
                if mode == "offline":
                    assert all(row["cached"] for row in result["runtimeInputs"])
                    assert set(result["cachedPackPayloads"].values()) == {None, str(cached)}
                else:
                    assert all(value is None for value in result["cachedPackPayloads"].values())
        assert all(path.endswith(".json") for path, _ in requests)
        if mode.startswith("offline"):
            assert requests == []
        assert {p: p.read_bytes() if p.is_file() else None for p in tmp_path.rglob("*")} == before
        assert not root.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_managed_source_registry_plan_uses_target_distribution_versions(tmp_path: Path) -> None:
    import subprocess
    import sys

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_supervisor.environment_development import source_identity
    from dinkster_workers.pack_lock import PackLock
    from test_environment_artifacts import _artifact, _json, _mirror
    from test_installer import _serve_artifacts, _static_indexes, _static_release

    base, _, _ = _mirror(tmp_path / "mirror")
    target = tmp_path / "target"
    target.mkdir()
    (target / "base-manifest.json").write_bytes(_json(base))
    engine = {
        "format": "dinkster.engine/1",
        "source": "git",
        "kind": "source",
        "version": "0.7",
        "workerProtocol": base["workerProtocol"],
        "published": False,
        "reproducible": False,
        "wheels": [
            _artifact("dinkster_api-0.5-py3-none-any.whl", b"api", key="filename"),
            _artifact("dinkster_inference-0.1.2-py3-none-any.whl", b"inference", key="filename"),
        ],
    }
    identity = source_identity(engine)
    manifest = target / "engine-manifest.json"
    manifest.write_bytes(_json(engine))
    files: dict[str, bytes] = {}
    releases = []
    for version, minimum in (("1.0", "0.1.2"), ("2.0", "0.1.3")):
        entry = LockedPack("example-pack", version, artifact_digest(version.encode()), "acme", ())
        row = _static_release(entry, version.encode())
        row["requires"] = [f"dinkster-inference>={minimum}"]
        payload = PackLock("example-pack", "fixture-cpu", ">=3.12", (), (), ()).to_json().encode()
        path = "locks/" + artifact_digest(payload).replace(":", "/") + ".json"
        files["/static/" + path] = payload
        row["installLocks"] = {
            "fixture-cpu": {
                "baseId": base["id"],
                "artifactDigest": entry.artifact_digest,
                "digest": artifact_digest(payload),
                "size": len(payload),
                "url": path,
            }
        }
        releases.append(row)
    files.update(_static_indexes(releases))
    requests: list[tuple[str, str | None]] = []
    server = _serve_artifacts(files, requests)
    try:
        (target / "dinkster.lock").write_text(
            Lockfile.of(
                [
                    LockedPack(
                        "example-pack",
                        "1.0",
                        artifact_digest(b"1.0"),
                        "acme",
                        (),
                        source=f"registry:http://127.0.0.1:{server.server_port}",
                    )
                ]
            ).record_json()
        )
        registry = tmp_path / "registries.toml"
        registry.write_text(
            f'[registries.private]\nendpoint="http://127.0.0.1:{server.server_port}"\ndefault=true\n'
        )
        arguments = [
            sys.executable,
            "-m",
            "dinkster.manager",
            "--root",
            str(tmp_path / "project"),
            "--registries",
            str(registry),
            "plan-registry-sources",
            "--engine-manifest",
            str(manifest),
            "--engine-release",
            identity,
            "--installed-lock",
            str(target / "dinkster.lock"),
            "--variant",
            "fixture-cpu",
            "--base-id",
            base["id"],
            "--update",
            "example-pack",
        ]
        planned = subprocess.run(arguments, capture_output=True, text=True)
        assert planned.returncode == 0, planned.stderr
        result = json.loads(planned.stdout)
        assert result["sources"]["packs"][0]["version"] == "1.0"
        assert "dinkster-inference==0.1.2" in result["skippedPackVersions"][0]
        fresh = arguments.copy()
        installed = fresh.index("--installed-lock")
        del fresh[installed : installed + 2]
        refused = subprocess.run(fresh, capture_output=True, text=True)
        assert refused.returncode != 0 and "engine manifest digest mismatch" in refused.stderr
        (target / "installed-wheels.json").write_text(
            json.dumps(
                {
                    "wheels": [
                        {"filename": "dinkster_api-0.5-py3-none-any.whl"},
                        {"filename": "dinkster_inference-0.1.3-py3-none-any.whl"},
                    ]
                }
            )
        )
        planned = subprocess.run(arguments, capture_output=True, text=True)
        assert planned.returncode == 0, planned.stderr
        assert json.loads(planned.stdout)["sources"]["packs"][0]["version"] == "2.0"
        engine["version"] = "9.0"
        manifest.write_bytes(_json(engine))
        refused = subprocess.run(arguments, capture_output=True, text=True)
        assert (
            refused.returncode != 0 and "source engine manifest identity mismatch" in refused.stderr
        )
        assert all(path.endswith(".json") for path, _ in requests)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "mode",
    ["online", "offline", "offline-archive", "offline-missing", "offline-corrupt", "offline-lock"],
)
def test_setup_yes_builds_the_approved_registry_sources_with_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd, mode: str
) -> None:
    import hashlib
    import os
    import subprocess
    import sys

    from dinkster_registry import LockedPack, artifact_digest
    from dinkster_store import object_path
    from dinkster_supervisor.environment_artifacts import acquire, discover, runtime_inputs
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.pack_environment import fetch_wheel
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from dinkster_workers.pack_wheel import build_pack_archive
    from test_environment_runtime import free_port
    from test_installer import _serve_artifacts, _static_indexes, _static_release
    from test_pack_wheel import wheel_pack

    from tests.fixtures.environments import build as publisher

    original_tree, original_wheel = publisher.standalone_tree, publisher.wheel
    encode = publisher.canonical_json

    def standalone(root: Path) -> None:
        original_tree(root)
        python = root / ("python.exe" if os.name == "nt" else "bin/python3")
        site = Path(
            subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        site.mkdir(parents=True, exist_ok=True)
        (site / "fixture.pth").write_text(
            "\n".join(str(Path(path).resolve()) for path in sys.path if path) + "\n"
        )

    def canonical_json(document: dict) -> bytes:
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        return encode(document)

    def wheel(*args, **kwargs):
        if kwargs.get("engine"):
            kwargs["pack_tools"] = True
        return original_wheel(*args, **kwargs)

    monkeypatch.setattr(publisher, "standalone_tree", standalone)
    monkeypatch.setattr(publisher, "canonical_json", canonical_json)
    monkeypatch.setattr(publisher, "wheel", wheel)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA", "DINKSTER_HOME"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    port = free_port()
    fixture = publisher.build(tmp_path / "release", "fixture-cpu", port)
    mirror, root, store = (Path(fixture[key]) for key in ("mirror", "root", "store"))
    monkeypatch.setenv("DINKSTER_HOME", str(root.parent))
    channel_path = mirror / "channels/nightly.json"
    channel = read_channel(channel_path)
    release = mirror / "engine" / channel["engine"]["release"][7:]
    engine = json.loads((release / "release-manifest.json").read_bytes())
    engine["defaultPacks"] = {"demo-nodes": "1.2.3"}
    data = canonical_json(engine)
    identity = hashlib.sha256(data).hexdigest()
    (release / "release-manifest.json").write_bytes(data)
    release.rename(release.parent / identity)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"][0]["path"] = f"engine/{identity}/release-manifest.json"
    channel_path.write_bytes(channel_json(channel))
    source = wheel_pack(tmp_path / "pack")
    (source / "locales/en.json").write_text("{}\n")
    manifest = source / "dinkster-pack.toml"
    manifest.write_text(
        manifest.read_text().replace('requires = ["demo-core>=2"]', "requires = []")
    )
    build_pack_archive(source, tmp_path / "pack-wheels")
    pack = tmp_path / "pack-wheels/demo_nodes-1.2.3-py3-none-any.whl"
    payload = pack.read_bytes()
    entry = LockedPack("demo-nodes", "1.2.3", artifact_digest(payload), "demo", ())
    row = _static_release(entry, payload)
    row["engine"] = ">=0.0.2"
    dependency = publisher.wheel(tmp_path / "pack-wheels", "demo_dependency", "2.0", 19)
    dependency_bytes = dependency.read_bytes()
    dependency_wheel = LockedWheel(
        dependency.name,
        len(dependency_bytes),
        hashlib.sha256(dependency_bytes).hexdigest(),
        dependency.as_uri(),
    )
    lock = PackLock(
        "demo-nodes",
        "fixture-cpu",
        ">=3.12",
        ("demo-dependency==2.0",),
        (),
        (
            LockedDistribution(
                "demo-dependency",
                "2.0",
                (
                    LockedWheel(
                        "demo_dependency-2.0-cp310-cp310-win32.whl",
                        17,
                        "a" * 64,
                        "https://example.invalid/unselected.whl",
                    ),
                    dependency_wheel,
                ),
            ),
            LockedDistribution(
                "fixture-base",
                "1.0",
                (
                    LockedWheel(
                        "fixture_base-1.0-py3-none-any.whl",
                        17,
                        "b" * 64,
                        "https://example.invalid/inherited.whl",
                    ),
                ),
            ),
        ),
    )
    lock_bytes = lock.to_json().encode()
    lock_digest = artifact_digest(lock_bytes)
    lock_path = "locks/" + lock_digest.replace(":", "/") + ".json"
    base = json.loads(Path(fixture["baseManifest"]).read_bytes())
    row["installLocks"] = {
        "fixture-cpu": {
            "baseId": base["id"],
            "artifactDigest": entry.artifact_digest,
            "digest": lock_digest,
            "size": len(lock_bytes),
            "url": lock_path,
        }
    }
    wheel_path = "/static/" + cast("list[dict[str, str]]", row["wheels"])[0]["url"]
    files = {
        **_static_indexes([row]),
        "/static/" + lock_path: lock_bytes,
        wheel_path: payload,
    }
    requests = []
    server = _serve_artifacts(files, requests)
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        cached_pack: Path | None = None
        missing_runtime: Path | None = None
        if mode != "online":
            from dinkster.registries import (
                NamedRegistry,
                RegistryConfig,
                StaticPackIndex,
                plan_registry_sources,
            )

            catalog = discover((str(mirror),), "nightly")
            acquire(catalog, "fixture-cpu", store)
            registry = NamedRegistry("fixture", endpoint, default=True)
            _, locks = plan_registry_sources(
                RegistryConfig((registry,)),
                registry,
                {"demo-nodes": "1.2.3"},
                engine_version=engine["version"],
                variant="fixture-cpu",
                base_id=base["id"],
                index=StaticPackIndex(store / "metadata/registry"),
            )
            cached_pack = fetch_wheel(locks[0].packages[-1].wheels[0], store / "wheels")
            cached_dependency = fetch_wheel(dependency_wheel, store / "wheels")
            missing_runtime = next(
                Path(cast("str", item["path"]))
                for item in runtime_inputs(catalog, "fixture-cpu", store)
                if cast("str", item["path"]).endswith(".whl")
            )
            if mode == "offline-missing":
                cached_pack.unlink()
                cached_dependency.unlink()
                missing_runtime.unlink()
            elif mode == "offline-corrupt":
                cached_pack.write_bytes(b"corrupt pack")
                cached_dependency.write_bytes(b"corrupt dependency")
                missing_runtime.write_bytes(b"corrupt runtime")
            elif mode == "offline-lock":
                for metadata in (store / "metadata/registry").rglob("*"):
                    if metadata.is_file() and metadata.read_bytes() == lock_bytes:
                        metadata.unlink()
                missing_runtime.unlink()
            elif mode == "offline-archive":
                import shutil

                shutil.rmtree(object_path(store / "bases", base["id"]))
            files.clear()
            mirror.rename(tmp_path / "disconnected")
            requests.clear()
            for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                monkeypatch.setenv(variable, "http://127.0.0.1:9")
            for variable in ("NO_PROXY", "no_proxy"):
                monkeypatch.setenv(variable, "")
            monkeypatch.setenv("UV_OFFLINE", "1")
            monkeypatch.setattr("urllib.request._opener", None)
        before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
        if mode in {"offline-missing", "offline-corrupt", "offline-lock"}:
            assert (
                cli_main(
                    [
                        "setup",
                        "--root",
                        str(root),
                        "--data",
                        str(store),
                        "--variant",
                        "fixture-cpu",
                        "--channel",
                        "nightly",
                        "--mirror",
                        str(mirror),
                        "--registry",
                        endpoint,
                        "--port",
                        str(port),
                        "--yes",
                        "--offline",
                    ]
                )
                == 1
            )
            error = capfd.readouterr().err.split("setup refused: ", 1)[-1]
            assert error.startswith("offline setup inputs missing or corrupt:")
            assert cached_pack is not None and missing_runtime is not None
            assert missing_runtime.name in error
            if mode == "offline-lock":
                assert "dependency lock" in error
            else:
                assert cached_pack.name in error
                assert dependency.name in error
            assert {p: p.read_bytes() for p in store.rglob("*") if p.is_file()} == before
            assert not root.exists()
            assert requests == []
            return
        # The registry must already be listening when choosing the supervisor's port.
        port = free_port()
        code = cli_main(
            [
                "setup",
                "--data",
                str(store),
                "--variant",
                "fixture-cpu",
                "--channel",
                "nightly",
                "--mirror",
                str(mirror),
                "--registry",
                endpoint,
                "--port",
                str(port),
                "--yes",
                "--json",
                *(["--offline"] if mode != "online" else []),
            ]
        )
        captured = capfd.readouterr()
        assert code == 0, captured.out + captured.err
        assert (root / "current").read_text().strip() == "1"
        selected = json.loads((root / "builds/1/dinkster.lock").read_text())
        assert selected["packs"][0]["artifactDigest"] == entry.artifact_digest
        retained = PackLock.read(root / "builds/1/packs/demo-nodes/pack-lock.json")
        assert retained.pack == "demo-nodes"
        assert retained.packages[-1].wheels[0].sha256 == hashlib.sha256(payload).hexdigest()
        if mode != "online":
            assert requests == []
        else:
            assert any(path == wheel_path for path, _ in requests)
    finally:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dinkster_supervisor.environment",
                "--root",
                str(root),
                "stop",
                "--yes",
                "--json",
            ],
            check=False,
            capture_output=True,
        )
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("existing", ["new", "alternate", "offline"])
def test_helper_default_mounts_preserve_policy_and_do_not_restore_removed_grants(
    tmp_path: Path, existing: str
) -> None:
    library = tmp_path / "library"
    configured = MountDef(
        "renders" if existing == "alternate" else "output",
        tmp_path / "unavailable-output",
        "readwrite" if existing == "alternate" else "read",
        priority=7,
    )
    with LibraryClient(library) as helper:
        if existing != "new":
            MountConfiguration.load(helper).save(
                (configured,), output_mount="renders" if existing == "alternate" else None
            )
        setup.ensure_default_mounts(library, helper)
        grants = MountConfiguration.load(helper)
        if existing == "new":
            assert set(grants.mounts) == {
                MountDef("output", library / "output", "readwrite"),
                MountDef("imports", library / "imports", "readwrite"),
            }
            assert grants.output_mount == "output"
        else:
            assert set(grants.mounts) == {
                configured,
                MountDef("imports", library / "imports", "readwrite"),
            }
            assert grants.output_mount == ("renders" if existing == "alternate" else None)
            assert not configured.path.exists()
            assert not (library / "output").exists()
        projects = ProjectFolders(helper, shared_grants=True)
        available = tuple(mount.id for mount in grants.mounts)
        assert set(projects.ordered("default", available)) == set(available)
        projects.update(
            "film",
            [mount for mount in available if mount != "imports"],
            available=available,
            output=None,
        )
        remaining = tuple(mount for mount in grants.mounts if mount.id != "imports")
        latest = MountConfiguration.load(helper)
        latest.save(remaining, output_mount=latest.output_mount)
        preferences = helper.call("folders.projects.get")
        (library / "imports").rmdir()
    with LibraryClient(library) as reopened:
        setup.ensure_default_mounts(library, reopened)
        assert MountConfiguration.load(reopened).mounts == remaining
        assert reopened.call("folders.projects.get") == preferences
        assert not (library / "imports").exists()
    assert not (library / "mounts.toml").exists()
    assert not (library / "project-folders.json").exists()


def test_setup_preserves_a_selected_alternate_output_mount(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library = tmp_path / "library"
    library.mkdir(parents=True)
    alternate = library / "renders"
    alternate.mkdir()
    mounts = (MountDef("renders", alternate, "readwrite"),)
    (library / "mounts.toml").write_text(dump_mounts(mounts, output_mount="renders"), "utf-8")

    setup.ensure_default_mounts(library)

    assert load_output_mount(library / "mounts.toml") == "renders"
    assert load_mounts(library / "mounts.toml") == (
        MountDef("imports", library / "imports", "readwrite"),
        *mounts,
    )
    assert not (library / "output").exists()


@pytest.mark.parametrize("custom_id", [False, True])
def test_setup_preserves_existing_offline_output_grants(
    tmp_path: Path, monkeypatch, custom_id: bool
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library = tmp_path / "library"
    library.mkdir(parents=True)
    root = library / "output" if custom_id else tmp_path / "offline-output"
    existing = MountDef("renders" if custom_id else "output", root, "read", priority=7)
    config = library / "mounts.toml"
    config.write_text(dump_mounts((existing,)), "utf-8")

    setup.ensure_default_mounts(library)
    saved = config.read_bytes()
    setup.ensure_default_mounts(library)

    assert config.read_bytes() == saved
    assert existing in load_mounts(config)
    assert load_output_mount(config) is None
    assert not root.exists()
    assert not (library / "output").exists()


def test_setup_does_not_restore_a_removed_output_grant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library = tmp_path / "library"
    setup.ensure_default_mounts(library)
    config = library / "mounts.toml"
    remaining = tuple(mount for mount in load_mounts(config) if mount.id != "output")
    config.write_text(dump_mounts(remaining), "utf-8")
    (library / "output").rmdir()

    setup.ensure_default_mounts(library)

    assert load_mounts(config) == remaining
    assert load_output_mount(config) is None
    assert not (library / "output").exists()


@pytest.mark.parametrize("custom_id", [False, True])
def test_setup_preserves_existing_import_grants(
    tmp_path: Path, monkeypatch, custom_id: bool
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library = tmp_path / "library"
    library.mkdir(parents=True)
    root = library / "imports" if custom_id else tmp_path / "offline-network-imports"
    existing = MountDef("plates" if custom_id else "imports", root, "read", priority=7)
    config = library / "mounts.toml"
    config.write_text(dump_mounts((existing,)), "utf-8")
    project_path = library / "project-folders.json"
    projects = ProjectFolders(project_path)
    projects.initialize((existing.id,))
    projects.register(existing.id, project="film", shared=False)
    projects.update("other", [], available=(existing.id,), output=None)
    preferences = json.loads(project_path.read_text("utf-8"))

    setup.ensure_default_mounts(library)
    saved = config.read_bytes()
    setup.ensure_default_mounts(library)

    assert config.read_bytes() == saved
    assert existing in load_mounts(config)
    assert len(load_mounts(config)) == 2
    assert not root.exists()
    assert not (library / "imports").exists()
    assert load_output_mount(config) == "output"
    assert json.loads(project_path.read_text("utf-8")) == {**preferences, "shared": ["output"]}


def test_setup_does_not_restore_a_removed_import_grant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library = tmp_path / "library"
    setup.ensure_default_mounts(library)
    config = library / "mounts.toml"
    remaining = tuple(mount for mount in load_mounts(config) if mount.id != "imports")
    config.write_text(dump_mounts(remaining, output_mount="output"), "utf-8")
    (library / "imports").rmdir()

    setup.ensure_default_mounts(library)

    assert load_mounts(config) == remaining
    assert not (library / "imports").exists()


@pytest.mark.parametrize("interrupt_settings_write", (False, True))
def test_setup_adds_imports_to_existing_projects_without_resetting_preferences(
    tmp_path: Path, monkeypatch, interrupt_settings_write: bool
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "state"))
    library = tmp_path / "library"
    library.mkdir(parents=True)
    config = library / "mounts.toml"
    mounts = (
        MountDef("output", library / "output", "readwrite"),
        MountDef("models", tmp_path / "models"),
    )
    config.write_text(dump_mounts(mounts, output_mount="output"), "utf-8")
    project_path = library / "project-folders.json"
    projects = ProjectFolders(project_path)
    projects.initialize(("output", "models"))
    projects.update("film", ["models"], available=("output", "models"), output=None)
    projects.update("music", ["output", "models"], available=("output", "models"), output="output")
    before = json.loads(project_path.read_text("utf-8"))

    if interrupt_settings_write:
        replace = setup.os.replace

        def interrupt(source: str | Path, destination: str | Path) -> None:
            if Path(destination) == project_path:
                raise OSError("project settings could not be published")
            replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(setup.os, "replace", interrupt)
            with pytest.raises(OSError, match="project settings could not be published"):
                setup.ensure_default_mounts(library)
        assert any(mount.id == "imports" for mount in load_mounts(config))
        assert json.loads(project_path.read_text("utf-8")) == before
        assert not (library / ".imports-initialized").exists()

    setup.ensure_default_mounts(library)

    after = json.loads(project_path.read_text("utf-8"))
    assert after == {**before, "shared": ["output", "models", "imports"]}
    table = MountTable(output_mount="output")
    for mount in load_mounts(config):
        table.add(mount)
    service = MountService(table, config)
    assert service.project_context("film").mounts == ("models", "imports")
    assert service.project_context("film").output_mount is None
    assert service.project_context("music").mounts == ("output", "models", "imports")
    assert service.project_context("music").output_mount == "output"
    service.projects.update("film", ["models"], available=service.mount_ids(), output=None)
    saved = project_path.read_bytes()

    setup.ensure_default_mounts(library)

    assert project_path.read_bytes() == saved
    restarted = MountService(table, config)
    assert restarted.project_context("film").mounts == ("models",)
    assert restarted.project_context("music").mounts == ("output", "models", "imports")


@pytest.mark.parametrize("arguments", [["serve"], ["serve-script"]])
def test_public_launch_serves_the_active_generation_and_stops_it(
    tmp_path: Path, monkeypatch, capsys, arguments: list[str]
) -> None:
    from urllib.request import urlopen

    from test_environment_cli import _runtime_project

    root = tmp_path / "state/project"
    root.mkdir(parents=True)
    port = _runtime_project(root)
    monkeypatch.setenv("DINKSTER_HOME", str(root.parent))
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    try:
        if arguments == ["serve-script"]:
            monkeypatch.setattr("sys.argv", ["dinkster-serve"])
            assert launch.serve_main() == 0
        else:
            assert cli_main(arguments) == 0
        assert opened == []
        assert f"http://127.0.0.1:{port}" in capsys.readouterr().out
        with urlopen(f"http://127.0.0.1:{port}/api/generation", timeout=5) as response:
            assert json.load(response) == {"generation": str(root / "builds/1")}
    finally:
        assert cli_main(["stop", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["engines"][0]["state"] == "stopped"


def test_dinkster_opens_the_installation_behind_the_always_on_process(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import os
    import signal
    from urllib.request import urlopen

    from test_environment_cli import _runtime_project
    from test_environment_runtime import free_port

    root = tmp_path / "home/project"
    root.mkdir(parents=True)
    _runtime_project(root)
    store = tmp_path / "store"
    store.mkdir()
    (store / "projects.json").write_text(
        json.dumps({"format": "dinkster.store/1", "projects": [str(root)]})
    )
    (tmp_path / "data/dinkster").mkdir(parents=True)
    (tmp_path / "data/dinkster/stores.json").write_text(
        json.dumps({"format": "dinkster.stores/1", "volumes": {"1": str(store)}})
    )
    monkeypatch.setenv("DINKSTER_HOME", str(root.parent))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<title>Dinkster</title>")
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    url = f"http://127.0.0.1:{free_port()}"

    def status() -> dict:
        with urlopen(url + "/supervisor/status", timeout=5) as response:
            return json.load(response)

    try:
        # With no installation the page still opens; nothing is started behind it.
        empty = ["--root", str(tmp_path / "absent"), "--port", url.rsplit(":", 1)[1]]
        assert cli_main([*empty, "--frontend-root", str(frontend)]) == 0
        assert opened == [url] and url in capsys.readouterr().out
        assert status()["state"] == "stopped" and "open" not in status()
        with urlopen(url + "/", timeout=5) as response:
            assert b"<title>Dinkster</title>" in response.read()

        # A second run reuses the running process and opens the installation in it.
        assert cli_main(["--no-browser", "--port", url.rsplit(":", 1)[1]]) == 0
        assert opened == [url]
        deadline = time.monotonic() + 60
        while (current := status())["state"] == "starting" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert current["state"] == "ready", current
        assert current["open"] == {"root": str(root.resolve()), "engine": "main"}
        with urlopen(url + "/api/generation", timeout=5) as response:
            assert json.load(response) == {"generation": str(root / "builds/1")}
        capsys.readouterr()

        # Stopping only the engine leaves Dinkster up and reads as stopped, not failed.
        assert cli_main(["stop", "--engine-only", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["engines"][0]["state"] == "stopped"
        deadline = time.monotonic() + 30
        while (current := status())["state"] == "ready" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert current["state"] == "stopped" and "detail" not in current, current

        # Plain stop shuts down the process and every engine it opened.
        assert cli_main(["--no-browser", "--port", url.rsplit(":", 1)[1]]) == 0
        deadline = time.monotonic() + 60
        while (current := status())["state"] != "ready" and time.monotonic() < deadline:
            time.sleep(0.05)
        capsys.readouterr()
        assert cli_main(["stop", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == {
            "status": "stopped",
            "stopped": [{"root": str(root.resolve()), "engine": "main"}],
            "failed": [],
        }
        # Once stop reports, the process can no longer be found.
        assert not (root.parent / "home.json").exists()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                status()
            except OSError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("the process kept answering after dinkster stop")
        assert cli_main(["stop", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["engines"][0]["state"] == "stopped"
    finally:
        with contextlib.suppress(OSError, KeyError):
            os.kill(status()["supervisor"]["pid"], signal.SIGTERM)
        cli_main(["stop", "--engine-only", "--json"])
        capsys.readouterr()


@pytest.mark.parametrize(
    "command", [["serve"], ["stop"], ["update"], ["restore-previous"], ["rollback"]]
)
def test_public_commands_without_setup_refuse_without_creating_state(
    tmp_path: Path, monkeypatch, capsys, command: list[str]
) -> None:
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "absent"))
    assert cli_main([*command, "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("command", [["serve"], ["stop"]])
def test_public_runtime_plan_leaves_the_project_unchanged(
    tmp_path: Path, monkeypatch, capsys, command: list[str]
) -> None:
    from test_environment_cli import _runtime_project

    root = tmp_path / "project"
    root.mkdir()
    _runtime_project(root)
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "wrong-default"))
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert cli_main([*command, "--root", str(root), "--plan", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "planned" and result["current"] == 1
    assert result["engines"] == []
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert not (tmp_path / "wrong-default").exists()


@pytest.mark.parametrize("restore_command", ["restore-previous", "rollback"])
def test_public_update_and_rollback_rebuild_the_selected_project(
    tmp_path: Path, monkeypatch, capfd, restore_command: str
) -> None:
    import subprocess

    from test_environment_runtime import free_port

    from tests.fixtures.environments.build import build

    fixture = build(tmp_path / "fixture", "fixture-cpu", free_port())
    root = Path(fixture["root"])
    for name in ("XDG_DATA_HOME", "LOCALAPPDATA", "DINKSTER_HOME"):
        monkeypatch.setenv(name, str(tmp_path / "user-data"))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    try:
        run_with_diagnostics(fixture["commands"]["setup"], timeout=60).check_returncode()
        assert cli_main(["stop", "--root", str(root), "--json"]) == 0
        capfd.readouterr()
        subprocess.run(fixture["commands"]["advance"], check=True, capture_output=True, timeout=30)
        for operation, expected_version in (("update", "2.0"), (restore_command, "1.0")):
            current = (root / "current").read_bytes()
            assert cli_main([operation, "--root", str(root), "--plan", "--json"]) == 0
            assert json.loads(capfd.readouterr().out)["status"] == "planned"
            assert (root / "current").read_bytes() == current
            assert cli_main([operation, "--root", str(root), "--yes", "--json"]) == 0
            result = json.loads(capfd.readouterr().out)
            assert result["status"] == "applied", result
            generation = root / "builds" / str(result["current"])
            assert (
                json.loads((generation / "build.json").read_text())["engine"]["version"]
                == expected_version
            )
            assert cli_main(["stop", "--root", str(root), "--json"]) == 0
            capfd.readouterr()
        assert not (tmp_path / "user-data/project").exists()
    finally:
        cli_main(["stop", "--root", str(root), "--json"])


def test_static_frontend_preserves_api_routes_and_spa_fallback(tmp_path: Path, monkeypatch) -> None:
    from dinkster_server.app import install_browser_request_security, install_cors

    system_guess_type = frontend.mimetypes.guess_type

    def windows_guess_type(name: str) -> tuple[str | None, str | None]:
        if name.endswith(".js"):
            return "application/javascript", None
        return system_guess_type(name)

    monkeypatch.setattr(frontend.mimetypes, "guess_type", windows_guess_type)

    async def scenario() -> None:
        bundle = tmp_path / "dist"
        (bundle / "assets").mkdir(parents=True)
        (bundle / "index.html").write_text("<main>editor</main>")
        (bundle / "assets" / "app.js").write_text("console.log('app')")
        app = web.Application()
        origin = "https://editor.example"
        install_browser_request_security(app, allow_hosts=(), allow_origins=(origin,))
        install_cors(app, (origin,))

        async def health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        app.router.add_get("/api/health", health)
        install_frontend(app, bundle=bundle)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert await (await client.get("/api/health")).json() == {"ok": True}
            assert (await client.head("/api/health")).status == 200
            for path in ("/workflow/one", "/project/one", "/apiary"):
                page = await client.get(path)
                assert page.status == 200
                assert await page.text() == "<main>editor</main>"
            for path in ("/api", "/api/", "/api/compat/comfy/schemas?version=1"):
                for method in ("GET", "HEAD", "POST"):
                    response = await client.request(method, path)
                    assert response.status == 404
                    assert response.content_type != "text/html"
                    if method == "HEAD":
                        assert await response.read() == b""
            asset = await client.get("/assets/app.js", headers={"Origin": origin})
            assert asset.content_type == "text/javascript"
            assert await asset.text() == "console.log('app')"
            assert asset.headers["Access-Control-Allow-Origin"] == origin
            assert (await client.head("/assets/app.js")).content_type == "text/javascript"
            refused = await client.get("/assets/app.js", headers={"Host": "untrusted.example"})
            assert refused.status == 421
            assert (await client.post("/unknown")).status == 404
        finally:
            await client.close()

    run_scenario(scenario())


def test_frontend_dev_proxies_http_and_websockets() -> None:
    import gzip

    async def scenario() -> None:
        vite = web.Application()
        compressed = gzip.compress(b"upstream compressed frontend bytes" * 100)

        async def http(request: web.Request) -> web.Response:
            if request.path == "/compressed":
                return web.Response(
                    body=compressed,
                    content_type="text/javascript",
                    headers={"Content-Encoding": "gzip"},
                )
            return web.Response(text=f"{request.method} {request.path_qs} {await request.text()}")

        async def websocket(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            async for message in socket:
                await socket.send_str(f"vite:{message.data}")
            return socket

        async def redirect(_request: web.Request) -> web.Response:
            raise web.HTTPFound("/hmr")

        vite.router.add_get("/redirect-hmr", redirect)
        vite.router.add_get("/hmr", websocket)
        vite.router.add_route("*", "/{path:.*}", http)
        vite_server = TestServer(vite, host="localhost")
        await vite_server.start_server()
        app = web.Application()

        async def health(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True})

        app.router.add_get("/api/health", health)
        install_frontend(
            app, development_url=str(vite_server.make_url("/")).replace("localhost", "LOCALHOST")
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert await (await client.get("/api/health")).json() == {"ok": True}
            assert (await client.head("/api/health")).status == 200
            for path in ("/api", "/api/", "/api/compat/comfy/schemas"):
                for method in ("GET", "HEAD", "POST"):
                    response = await client.request(method, path)
                    assert response.status == 404
                    if method == "HEAD":
                        assert await response.read() == b""
            for path in ("/project/one", "/apiary"):
                response = await client.get(path)
                assert response.status == 200
                assert await response.text() == f"GET {path} "
            compressed_response = await client.get("/compressed", auto_decompress=False)
            assert compressed_response.headers["Content-Encoding"] == "gzip"
            assert await compressed_response.read() == compressed
            response = await client.post("/asset?version=2", data="body")
            assert await response.text() == "POST /asset?version=2 body"
            for path in ("/hmr", "/redirect-hmr"):
                socket = await client.ws_connect(path)
                await socket.send_str("change")
                assert (await socket.receive()).data == "vite:change"
                await socket.close()
        finally:
            await client.close()
            await vite_server.close()

    run_scenario(scenario())


def test_frontend_dev_rejected_redirect_closes_unfinished_response() -> None:
    import asyncio

    import aiohttp

    async def scenario() -> None:
        disconnected = asyncio.Event()

        async def redirect(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(
                status=302,
                headers={"Location": "http://127.0.0.1:9/outside", "Content-Length": "1000"},
            )
            await response.prepare(request)
            await response.write(b"unfinished")
            transport = request.transport
            try:
                for _ in range(500):
                    if transport is None or transport.is_closing():
                        disconnected.set()
                        break
                    await asyncio.sleep(0.01)
            finally:
                if transport is None or transport.is_closing():
                    disconnected.set()
            return response

        async def healthy(_request: web.Request) -> web.Response:
            return web.Response(text="still available")

        vite = web.Application()
        vite.router.add_get("/redirect", redirect)
        vite.router.add_get("/", healthy)
        vite_server = TestServer(vite)
        await vite_server.start_server()
        app = web.Application()
        install_frontend(app, development_url=str(vite_server.make_url("/")))
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for _ in range(3):
                disconnected.clear()
                with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
                    await client.ws_connect("/redirect")
                assert refused.value.status == 502
                await asyncio.wait_for(disconnected.wait(), timeout=2)
                assert await (await client.get("/")).text() == "still available"
        finally:
            await client.close()
            await vite_server.close()

    run_scenario(scenario())
