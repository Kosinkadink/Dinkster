"""Discovery and acquisition consume the publishers' existing manifest formats."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from platform import machine
from types import SimpleNamespace
from typing import cast

import pytest
import zstandard
from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_store import object_path, pack_archive_path
from dinkster_supervisor import environment_artifacts
from dinkster_supervisor.environment_artifacts import (
    acquire,
    acquire_pack_locks,
    acquire_sources,
    discover,
    runtime_inputs,
)
from dinkster_supervisor.environment_channels import write_channel
from dinkster_supervisor.environment_custom import custom_selection
from dinkster_supervisor.environment_downloads import DownloadError
from dinkster_supervisor.environment_state import EnvironmentError, content_digest
from test_environment_generations import _standalone_tree

from dinkster.generation_packs import stage_generation_packs


def _json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _artifact(name: str, data: bytes, *, key: str = "name") -> dict:
    return {key: name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _wheel_payload(name: str, version: str, requirement: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            f"Requires-Dist: {requirement}\n",
        )
    return output.getvalue()


def _mirror(
    root: Path,
    *,
    protocol: int = 11,
    damaged_archive: bool = False,
    variant: str = "fixture-cpu",
    target: tuple[str, str] | None = None,
) -> tuple[dict, dict, dict]:
    root.mkdir(parents=True)
    system, arch = target or (
        "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux",
        {"amd64": "x86_64", "aarch64": "arm64"}.get(machine().lower(), machine().lower()),
    )
    python = "python/python.exe" if system == "windows" else "python/bin/python3"
    base_wheel = b"base wheel bytes; execution is covered by generation tests"
    wheel = {
        **_artifact("fixture_base-1.0-py3-none-any.whl", base_wheel, key="filename"),
        "name": "fixture-base",
        "version": "1.0",
        "lockedVersion": "1.0",
        "index": "https://pypi.org/simple",
        "url": "https://example.org/fixture_base-1.0-py3-none-any.whl",
        "verification": "lock",
    }
    interpreter = {
        **_artifact("interpreter.tar.gz", b"unused source archive", key="filename"),
        "release": "fixture",
        "url": "https://example.org/interpreter.tar.gz",
        "executable": python,
    }
    base = {
        "format": "dinkster.base/1",
        "variant": variant,
        "platform": {"os": system, "arch": arch},
        "python": "3.14.10",
        "interpreter": interpreter,
        "wheels": [wheel],
        "replaces": ["fixture-base", "nvidia-cublas"],
        "workerProtocol": protocol,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    base_id = hashlib.sha256(_json(base)).hexdigest()
    tree = root / "fixture-tree"
    _standalone_tree(tree / "python")
    (tree / "wheels").mkdir()
    (tree / "wheels" / wheel["filename"]).write_bytes(b"wrong" if damaged_archive else base_wheel)
    compressed = io.BytesIO()
    with zstandard.ZstdCompressor().stream_writer(compressed, closefd=False) as encoded:
        with tarfile.open(fileobj=encoded, mode="w|", dereference=False) as archive:
            archive.add(tree / "python", "python")
            archive.add(tree / "wheels", "wheels")
    data = compressed.getvalue()
    shutil.rmtree(tree)
    base_path = root / f"base/{variant}/{base_id}"
    base_path.mkdir(parents=True)
    parts = (data[: len(data) // 2], data[len(data) // 2 :])
    base["archive"]["parts"] = []
    for index, part in enumerate(parts, 1):
        name = f"base.tar.zst.{index:03}"
        (base_path / name).write_bytes(part)
        base["archive"]["parts"].append(_artifact(name, part, key="filename"))
    base["id"], base["signature"] = "sha256:" + base_id, None
    (base_path / "manifest.json").write_bytes(_json(base) + b"\n")
    engine_wheel = b"engine wheel bytes"
    constraints = (
        "fixture-base==1.0 --hash=sha256:" + "a" * 64 + "\n"
        "nvidia-cublas==13.0 --hash=sha256:" + "c" * 64 + "\n"
        "fixture-engine==2.0 --hash=sha256:" + hashlib.sha256(engine_wheel).hexdigest() + "\n"
    ).encode("ascii")
    engine = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "tag": "main-" + "d" * 40,
        "commit": "d" * 40,
        "version": "0.4.0.dev1",
        "workerProtocol": 11,
        "artifacts": [
            _artifact("constraints.txt", constraints),
            _artifact("fixture_engine-2.0-py3-none-any.whl", engine_wheel),
        ],
    }
    engine_data = _json(engine) + b"\n"
    release = hashlib.sha256(engine_data).hexdigest()
    engine_path = root / f"engine/{release}"
    engine_path.mkdir(parents=True)
    (engine_path / "release-manifest.json").write_bytes(engine_data)
    (engine_path / "constraints.txt").write_bytes(constraints)
    (engine_path / "fixture_engine-2.0-py3-none-any.whl").write_bytes(engine_wheel)
    channel = {
        "format": "dinkster.channel/1",
        "engine": {
            "version": engine["version"],
            "release": "sha256:" + release,
            "locations": [{"type": "mirror", "path": f"engine/{release}/release-manifest.json"}],
        },
        "base": {
            variant: {
                "id": base["id"],
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/{base_id}/manifest.json"}
                ],
            }
        },
        "minimumDesktopVersion": "0.2.0",
        "signature": None,
    }
    (root / "channels").mkdir()
    for name in ("stable", "main"):
        (root / f"channels/{name}.json").write_bytes(_json(channel))
    return base, engine, channel


def _release_sources(
    mirror: Path, *, dependency_locks: dict[str, bytes] | None = None
) -> tuple[Path, dict[str, bytes]]:
    channel_path = mirror / "channels/main.json"
    channel = json.loads(channel_path.read_bytes())
    previous = mirror / "engine" / channel["engine"]["release"][7:]
    engine = json.loads((previous / "release-manifest.json").read_bytes())
    entries, payloads = [], {}
    for name in ("first-pack", "second-pack"):
        source = mirror / name
        source.mkdir()
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'namespaces = ["{name}"]\n[pack.entry]\nnodes = "nodes:NODES"\n'
        )
        (source / "nodes.py").write_text(f'VALUE = "{name}"\n')
        archive = mirror / f"{name}-1.0.zip"
        identity = build_artifact(source, archive)
        entries.append(LockedPack(name, "1.0", identity, "dinkster", (name,)))
        payloads[archive.name] = archive.read_bytes()
    payloads["official-packs.lock"] = (Lockfile.of(entries).record_json() + "\n").encode()
    payloads.update(dependency_locks or {})
    engine["artifacts"].extend(_artifact(name, data) for name, data in payloads.items())
    data = _json(engine) + b"\n"
    identity = hashlib.sha256(data).hexdigest()
    release = mirror / "engine" / identity
    shutil.copytree(previous, release)
    (release / "release-manifest.json").write_bytes(data)
    for name, payload in payloads.items():
        (release / name).write_bytes(payload)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"] = [
        {"type": "mirror", "path": f"engine/{identity}/release-manifest.json"}
    ]
    channel_path.write_bytes(_json(channel))
    return release, payloads


def test_acquires_only_selected_variant_dependency_locks(tmp_path: Path):
    from dinkster_workers.pack_lock import PackLock

    mirror, store = tmp_path / "mirror", tmp_path / "store"
    _mirror(mirror)
    payloads = {}
    for pack, variant in (
        ("first-pack", "fixture-cpu"),
        ("second-pack", "fixture-cpu"),
        ("first-pack", "win-cpu"),
    ):
        path = tmp_path / f"{pack}-0.4.0.dev1-{variant}.pack-lock.json"
        PackLock(pack, variant, ">=3.12", (), (), ()).write(path)
        payloads[path.name] = path.read_bytes()
    release, _ = _release_sources(mirror, dependency_locks=payloads)
    (release / "first-pack-0.4.0.dev1-win-cpu.pack-lock.json").write_bytes(b"unselected corruption")
    catalog = discover((str(mirror),), "main")
    acquired = acquire_pack_locks(catalog, "fixture-cpu", store)
    assert {path.name for path in acquired} == {
        "first-pack-0.4.0.dev1-fixture-cpu.pack-lock.json",
        "second-pack-0.4.0.dev1-fixture-cpu.pack-lock.json",
    }
    for path in acquired:
        assert path.read_bytes() == payloads[path.name]
        assert PackLock.read(path).packages == ()
    assert not list(store.rglob("*win-cpu.pack-lock.json"))
    acquired[0].write_bytes(b"damaged cache")
    acquire_pack_locks(catalog, "fixture-cpu", store)
    assert acquired[0].read_bytes() == payloads[acquired[0].name]
    (release / acquired[0].name).write_bytes(b"damaged selected release")
    with pytest.raises(DownloadError):
        acquire_pack_locks(catalog, "fixture-cpu", tmp_path / "fresh-store")


def test_acquires_release_sources_into_the_existing_artifact_identity_cache(tmp_path: Path):
    acquired_date = datetime.now(UTC).date().isoformat()
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    _mirror(mirror)
    release, payloads = _release_sources(mirror)
    catalog = discover((str(mirror),), "main")
    with pytest.raises(EnvironmentError, match="intact cached release source"):
        acquire_sources(catalog, store, download=False)
    assert not store.exists()
    acquired = acquire_sources(catalog, store)
    assert acquired is not None
    lock, artifacts = acquired
    assert lock.read_bytes() == payloads["official-packs.lock"]
    entries = Lockfile.from_record_json(lock.read_text()).packs
    assert len(entries) == 2
    assert acquire_sources(catalog, store, download=False) == acquired
    for entry in entries:
        cached = pack_archive_path(artifacts, entry.artifact_digest)
        assert cached == artifacts / entry.pack / f"{acquired_date}-1.0" / "pack.zip"
        assert cached.read_bytes() == payloads[f"{entry.pack}-1.0.zip"]
        cached.write_bytes(b"damaged cache")
    with pytest.raises(EnvironmentError, match="intact cached release source"):
        acquire_sources(catalog, store, download=False)
    for entry in entries:
        cached = pack_archive_path(artifacts, entry.artifact_digest)
        assert cached.read_bytes() == b"damaged cache"
    acquire_sources(catalog, store)
    for entry in entries:
        cached = pack_archive_path(artifacts, entry.artifact_digest)
        assert cached.read_bytes() == payloads[f"{entry.pack}-1.0.zip"]
    target = tmp_path / "target"
    target.mkdir()
    stage_generation_packs(
        target,
        Lockfile.from_record_json(lock.read_text()),
        {entry.pack: pack_archive_path(artifacts, entry.artifact_digest) for entry in entries},
    )
    assert (target / "dinkster.lock").read_bytes() == payloads["official-packs.lock"]
    for entry in entries:
        assert (
            target / "pack-sources" / entry.pack / "nodes.py"
        ).read_text() == f'VALUE = "{entry.pack}"\n'
    (release / "first-pack-1.0.zip").write_bytes(b"damaged release")
    with pytest.raises(DownloadError, match="could not be downloaded"):
        acquire_sources(catalog, tmp_path / "fresh-store")
    assert not list((tmp_path / "fresh-store").rglob("*.zip"))


@pytest.mark.parametrize("channel_name", ["stable", "main"])
def test_discovers_intact_manifests_without_creating_cache_then_acquires_exact_bytes(
    tmp_path: Path,
    channel_name: str,
):
    mirror, store = tmp_path / "mirror", tmp_path / "data"
    base, engine, channel = _mirror(mirror)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    catalog = discover((str(mirror),), channel_name)
    assert catalog.document == channel
    assert catalog.engine == engine
    assert catalog.variants() == [
        {
            "variant": "fixture-cpu",
            "base": base["id"],
            "manifest": base,
            "downloadBytes": sum(row["size"] for row in base["archive"]["parts"]),
        }
    ]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    constraints = acquire(catalog, "fixture-cpu", store)
    assert "nvidia-cublas" not in constraints.read_text()
    assert (
        f"fixture-base==1.0 --hash=sha256:{base['wheels'][0]['sha256']}" in constraints.read_text()
    )
    acquired = sorted(store.glob("wheels/*/*.whl"))
    assert sorted(path.read_bytes() for path in acquired) == sorted(
        [b"base wheel bytes; execution is covered by generation tests", b"engine wheel bytes"]
    )
    for path in acquired:
        assert path.parent.name == path.stem
        assert object_path(store / "wheels", content_digest(path)) == path.parent
    retained = (
        object_path(store / "engines", channel["engine"]["release"]) / "release-manifest.json"
    )
    assert retained.read_bytes() == _json(engine) + b"\n"


@pytest.mark.parametrize("api", [1, 2, 4, 5])
def test_frontend_only_channel_update_keeps_engine_and_checks_api_before_acquisition(
    tmp_path: Path, api: int
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    _, engine, channel = _mirror(mirror)

    def frontend(version: str, minimum: int, maximum: int) -> dict:
        manifest = {
            "format": "dinkster.frontend/1",
            "repository": "Kosinkadink/Dinkster-Frontend",
            "version": version,
            "engineApi": {"min": minimum, "max": maximum},
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("dinkster_frontend/manifest.json", json.dumps(manifest))
            archive.writestr("dinkster_frontend/static/index.html", version)
        payload = output.getvalue()
        filename = f"dinkster_frontend-{version}-py3-none-any.whl"
        row = {
            **manifest,
            "url": f"https://files.pythonhosted.org/packages/{filename}",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        cached = object_path(store / "wheels", "sha256:" + row["sha256"]) / filename
        cached.parent.mkdir(parents=True)
        cached.write_bytes(payload)
        return row

    original, updated = frontend("0.1.0", 1, 5), frontend("0.2.0", 2, 4)
    old_release = mirror / "engine" / channel["engine"]["release"][7:]
    engine.update(schemaApi=api, frontend=original)
    data = _json(engine) + b"\n"
    identity = hashlib.sha256(data).hexdigest()
    release = mirror / "engine" / identity
    shutil.copytree(old_release, release)
    (release / "release-manifest.json").write_bytes(data)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"][0]["path"] = f"engine/{identity}/release-manifest.json"
    path = mirror / "channels/main.json"
    path.write_bytes(_json(channel))
    initial = discover((str(mirror),), "main")
    initial_constraints = acquire(initial, "fixture-cpu", store)
    requirements = initial_constraints.read_text()
    assert f"dinkster-frontend==0.1.0 --hash=sha256:{original['sha256']}" in requirements
    channel["frontend"] = updated
    path.write_bytes(_json(channel))
    before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    if api in {1, 5}:
        with pytest.raises(EnvironmentError, match="supports engine schema API 2 through 4"):
            discover((str(mirror),), "main")
        assert {p: p.read_bytes() for p in store.rglob("*") if p.is_file()} == before
        return
    selected = discover((str(mirror),), "main")
    assert selected.engine_bytes == initial.engine_bytes
    assert selected.channel.engine_release == initial.channel.engine_release
    requirements = acquire(selected, "fixture-cpu", store).read_text()
    assert initial_constraints.read_bytes() == before[initial_constraints]
    assert f"dinkster-frontend==0.2.0 --hash=sha256:{updated['sha256']}" in requirements
    assert "dinkster-frontend==0.1.0" not in requirements
    inputs = runtime_inputs(selected, "fixture-cpu", store)
    assert any(
        Path(cast("str", p["path"])).name == "dinkster_frontend-0.2.0-py3-none-any.whl"
        for p in inputs
    )
    assert not any(
        Path(cast("str", p["path"])).name == "dinkster_frontend-0.1.0-py3-none-any.whl"
        for p in inputs
    )


def test_discovery_reuses_verified_acquisition_metadata_without_network_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    base, engine, channel = _mirror(mirror)
    catalog = discover((str(mirror),), "main")
    constraints = acquire(catalog, "fixture-cpu", store).read_bytes()
    mirror.rename(tmp_path / "disconnected")
    before = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    with monkeypatch.context() as patch:
        read = environment_artifacts.read_metadata

        def cache_only(root: str, path: str) -> bytes:
            assert Path(root).is_relative_to(store / "metadata")
            return read(root, path)

        patch.setattr(environment_artifacts, "read_metadata", cache_only)
        offline = discover((str(mirror),), "main", store=store, offline=True)
        assert offline.document == channel
        assert offline.engine == engine
        assert offline.bases == {"fixture-cpu": base}
    cached = discover((str(mirror),), "main", store=store)
    assert cached.document == channel
    assert cached.engine == engine
    assert cached.bases == {"fixture-cpu": base}
    assert cached.cached_channel
    assert cached.engine_sources == catalog.engine_sources
    assert cached.base_sources == catalog.base_sources
    assert {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    } == before
    assert acquire(cached, "fixture-cpu", store).read_bytes() == constraints
    with pytest.raises(EnvironmentError, match="no mirror"):
        discover((str(tmp_path / "different-mirror"),), "main", store=store)
    (store / "metadata" / f"{base['id'][7:]}.json").write_text("{}")
    with pytest.raises(EnvironmentError, match="intact expected artifact manifest"):
        discover((str(mirror),), "main", store=store)
    (store / "metadata" / f"{channel['engine']['release'][7:]}.json").unlink()
    before = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    with monkeypatch.context() as patch:
        patch.setattr(environment_artifacts, "read_metadata", cache_only)
        with pytest.raises(EnvironmentError) as refused:
            discover((str(mirror),), "main", store=store, offline=True)
    assert channel["engine"]["release"] in str(refused.value)
    assert base["id"] in str(refused.value)
    assert {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize("cache", ["empty", "warm", "corrupt", "archive-only"])
def test_offline_runtime_acquisition_reports_all_missing_inputs_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache: str
):
    from dinkster_supervisor import environment_downloads

    mirror, store = tmp_path / "mirror", tmp_path / "store"
    base, engine, _ = _mirror(mirror)
    catalog = discover((str(mirror),), "main")
    if cache != "empty":
        acquire(catalog, "fixture-cpu", store)
    if cache == "corrupt":
        base_wheel = (
            object_path(store / "bases", base["id"]) / "wheels" / base["wheels"][0]["filename"]
        )
        engine_wheel = (
            object_path(store / "wheels", "sha256:" + engine["artifacts"][1]["sha256"])
            / engine["artifacts"][1]["name"]
        )
        base_wheel.write_bytes(b"damaged base")
        engine_wheel.write_bytes(b"damaged engine")
    elif cache == "archive-only":
        shutil.rmtree(object_path(store / "bases", base["id"]))

    def forbidden(*args, **kwargs):
        raise AssertionError("offline runtime acquisition must not contact a remote source")

    monkeypatch.setattr(environment_downloads, "build_opener", forbidden)
    monkeypatch.setattr(environment_artifacts, "dependency_source", forbidden)
    mirror.rename(tmp_path / "disconnected")
    before = {path: path.read_bytes() if path.is_file() else None for path in store.rglob("*")}
    inputs = runtime_inputs(catalog, "fixture-cpu", store)
    assert {
        path: path.read_bytes() if path.is_file() else None for path in store.rglob("*")
    } == before
    if cache in {"warm", "archive-only"}:
        assert all(row["cached"] for row in inputs)
        constraints = acquire(catalog, "fixture-cpu", store, offline=True)
        assert "fixture-engine==2.0" in constraints.read_text()
        assert "fixture-base==1.0" in constraints.read_text()
        assert (
            object_path(store / "bases", base["id"]) / "wheels" / base["wheels"][0]["filename"]
        ).is_file()
    else:
        with pytest.raises(
            EnvironmentError, match="offline runtime inputs missing or corrupt"
        ) as error:
            acquire(catalog, "fixture-cpu", store, offline=True)
        expected = (
            (
                "base.tar.zst.001",
                "base.tar.zst.002",
                "constraints.txt",
                "fixture_engine-2.0-py3-none-any.whl",
            )
            if cache == "empty"
            else ("fixture_base-1.0-py3-none-any.whl", "fixture_engine-2.0-py3-none-any.whl")
        )
        assert all(name in str(error.value) for name in expected)
        assert {
            path: path.read_bytes() if path.is_file() else None for path in store.rglob("*")
        } == before
        if cache == "empty":
            assert not store.exists()


def test_acquisition_retains_published_base_metadata_without_replacing_custom_provenance(
    tmp_path: Path,
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    base, _, _ = _mirror(mirror)
    base.update(
        label="2026-11-01-py3.14.10-torch2.14.1",
        builtAt="2026-11-01T00:00:00Z",
        torchVersion="2.14.1",
        torchIndex="https://pypi.org/simple",
        lockSource={
            "repository": "https://github.com/Kosinkadink/Dinkster-Environments",
            "commit": "f" * 40,
        },
    )
    (mirror / "base/fixture-cpu" / base["id"][7:] / "manifest.json").write_bytes(_json(base))
    catalog = discover((str(mirror),), "main")
    acquire(catalog, "fixture-cpu", store)
    retained = object_path(store / "bases", base["id"]) / "manifest.json"
    assert retained.parent == store / "bases/fixture-cpu/2026-11-01-py3.14.10-torchnone"
    assert json.loads(retained.read_bytes()) == base
    custom = _json({**base, "source": "user-defined", "build": {"sourceDigest": "a" * 64}})
    retained.write_bytes(custom)
    acquire(catalog, "fixture-cpu", store)
    assert retained.read_bytes() == custom
    retained.unlink()
    wheel = retained.parent / "wheels" / base["wheels"][0]["filename"]
    wheel.write_bytes(b"damaged cached base")
    with pytest.raises(EnvironmentError, match="extracted base wheel differs"):
        acquire(catalog, "fixture-cpu", store)
    assert not retained.exists()
    assert all(base["label"] not in part for path in store.rglob("*") for part in path.parts)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_acquisition_retains_original_discovery_bytes_without_refetch(
    tmp_path: Path, encoding: str
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    _, engine, channel = _mirror(mirror)
    previous = mirror / "engine" / channel["engine"]["release"][7:]
    del engine["tag"]
    engine["label"] = "2026-11-01-0.4.0.dev1"
    data = (json.dumps(engine, indent=3) + "\r\n").encode(encoding)
    identity = hashlib.sha256(data).hexdigest()
    release = mirror / "engine" / f"{engine['label']}-{identity[:8]}"
    shutil.copytree(previous, release)
    manifest = release / "release-manifest.json"
    manifest.write_bytes(data)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"] = [
        {"type": "mirror", "path": f"engine/{release.name}/release-manifest.json"}
    ]
    (mirror / "channels/main.json").write_bytes(_json(channel))
    catalog = discover((str(mirror),), "main")
    assert not store.exists()
    manifest.write_bytes(b"changed after discovery")
    acquire(catalog, "fixture-cpu", store)
    retained = object_path(store / "engines", "sha256:" + identity) / "release-manifest.json"
    assert retained.read_bytes() == data
    assert hashlib.sha256(retained.read_bytes()).hexdigest() == identity
    assert retained.parent == store / "engines" / engine["label"]
    retained.write_bytes(b"corrupt cache")
    acquire(catalog, "fixture-cpu", store)
    assert retained.read_bytes() == data


@pytest.mark.parametrize("source", ["origin", "index"])
@pytest.mark.parametrize("failure", [None, "hash", "size", "missing", "base-hash", "dependency"])
def test_locked_dependencies_use_exact_remote_bytes_and_reuse_the_shared_store(
    tmp_path: Path, source: str, failure: str | None
):
    mirror, store, packages = tmp_path / "release", tmp_path / "store", tmp_path / "packages"
    base, engine, channel = _mirror(mirror)
    packages.mkdir()
    filename = "third_party-7.2-py3-none-any.whl"
    payload = _wheel_payload(
        "third_party", "7.2", "fixture-base>=2" if failure == "dependency" else "fixture-base<2"
    )
    (packages / filename).write_bytes(payload)
    index = packages / "simple/third-party"
    index.mkdir(parents=True)
    (index / "index.html").write_text(
        f'<a href="../../{filename}#sha256=untrusted-index-hash">locked</a>'
        '<a href="../../third_party-9.0.tar.gz">source must not be used</a>'
    )
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(packages)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        release = mirror / "engine" / channel["engine"]["release"][7:]
        engine_filename = "fixture_engine-2.0-py3-none-any.whl"
        engine_payload = _wheel_payload("fixture_engine", "2.0", "fixture-base>=1")
        (release / engine_filename).write_bytes(engine_payload)
        environment_artifacts.artifact_rows(engine["artifacts"], "name")[1].update(
            _artifact(engine_filename, engine_payload)
        )
        fields = ("name", "version", "filename", "size", "sha256", "index", "url")
        base_row = {key: base["wheels"][0][key] for key in fields}
        dependency = {
            **_artifact(filename, payload, key="filename"),
            "name": "third-party",
            "version": "7.2",
            "index": origin + "/simple",
            "url": (origin if source == "origin" else "https://unreachable.invalid")
            + "/"
            + filename,
        }
        if failure == "base-hash":
            base_row["sha256"] = "a" * 64
        engine["dependencies"] = {
            "fixture-cpu": {"base": base["id"], "wheels": [base_row, dependency]}
        }
        data = _json(engine)
        (release / "release-manifest.json").write_bytes(data)
        identity = hashlib.sha256(data).hexdigest()
        release.rename(release.with_name(identity))
        channel["engine"]["release"] = "sha256:" + identity
        channel["engine"]["locations"] = [
            {"type": "mirror", "path": f"engine/{identity}/release-manifest.json"}
        ]
        (mirror / "channels/main.json").write_bytes(_json(channel))
        catalog = discover((str(mirror),), "main")
        package_index = origin + "/simple" if source == "index" else None
        if failure is None:
            inspected = runtime_inputs(catalog, "fixture-cpu", store)
            assert not store.exists()
            assert requests == []
            assert sum(cast("int", row["size"]) for row in inspected) == (
                sum(row["size"] for row in base["archive"]["parts"])
                + sum(row["size"] for row in engine["artifacts"])
                + len(payload)
            )
            with pytest.raises(EnvironmentError) as refused:
                acquire(catalog, "fixture-cpu", store, package_index=package_index, offline=True)
            assert filename in str(refused.value)
            assert "base.tar.zst.001" in str(refused.value)
            assert "fixture_engine-2.0-py3-none-any.whl" in str(refused.value)
            assert not store.exists()
            assert requests == []
        if failure == "hash":
            (packages / filename).write_bytes(b"X" * len(payload))
        elif failure == "size":
            (packages / filename).write_bytes(payload + b"extra")
        elif failure == "missing":
            (packages / filename).unlink()
        if failure is not None:
            with pytest.raises(
                (DownloadError, EnvironmentError),
                match="third-party requires fixture-base>=2; base provides fixture-base 1.0"
                if failure == "dependency"
                else None,
            ):
                acquire(catalog, "fixture-cpu", store, package_index=package_index)
            assert (
                object_path(store / "wheels", "sha256:" + dependency["sha256"]) / filename
            ).exists() == (failure == "dependency")
            assert not list(store.glob("engines/*/constraints-fixture-cpu.txt"))
            assert not any(".tar.gz" in path for path in requests)
            if failure == "base-hash":
                assert not store.exists()
            return
        constraints = acquire(catalog, "fixture-cpu", store, package_index=package_index)
        assert (
            object_path(store / "wheels", "sha256:" + dependency["sha256"]) / filename
        ).read_bytes() == payload
        assert set(constraints.read_text().splitlines()) == {
            "fixture-engine==2.0 --hash=sha256:" + hashlib.sha256(engine_payload).hexdigest(),
            "third-party==7.2 --hash=sha256:" + hashlib.sha256(payload).hexdigest(),
            "fixture-base==1.0 --hash=sha256:" + base_row["sha256"],
        }
        requested = list(requests)
        (packages / filename).unlink()
        (index / "index.html").unlink()
        assert (
            acquire(catalog, "fixture-cpu", store, package_index=package_index, offline=True)
            == constraints
        )
        assert requests == requested
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("target", "host", "architecture", "accepted"),
    [
        (("macos", "arm64"), "darwin", "arm64", True),
        (("macos", "arm64"), "darwin", "aarch64", True),
        (("macos", "arm64"), "darwin", "x86_64", False),
        (("macos", "arm64"), "win32", "AMD64", False),
        (("macos", "arm64"), "win32", "ARM64", False),
        (("macos", "arm64"), "linux", "x86_64", False),
        (("macos", "arm64"), "linux", "aarch64", False),
        (("macos", "x86_64"), "darwin", "arm64", False),
        (("windows", "x86_64"), "darwin", "arm64", False),
        (("linux", "x86_64"), "darwin", "arm64", False),
    ],
)
def test_acquisition_requires_manifest_os_and_architecture_to_match_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: tuple[str, str],
    host: str,
    architecture: str,
    accepted: bool,
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    base, _, _ = _mirror(
        mirror, variant="mac-arm64" if target[0] == "macos" else "fixture-cpu", target=target
    )
    catalog = discover((str(mirror),), "main")
    monkeypatch.setattr(environment_artifacts, "sys", SimpleNamespace(platform=host))
    monkeypatch.setattr(environment_artifacts, "machine", lambda: architecture)
    variant = next(iter(catalog.bases))
    if not accepted:
        with pytest.raises(EnvironmentError, match="operating system|architecture"):
            acquire(catalog, variant, store)
        assert not store.exists()
        return
    # Exercise host admission without trying to execute a foreign interpreter on the test host.
    cached_wheels = store / "bases" / base["id"][7:] / "wheels"
    cached_wheels.mkdir(parents=True)
    (cached_wheels / base["wheels"][0]["filename"]).write_bytes(
        b"base wheel bytes; execution is covered by generation tests"
    )
    constraints = acquire(catalog, variant, store)
    assert "fixture-base==1.0 --hash=sha256:" in constraints.read_text()
    assert "nvidia-cublas" not in constraints.read_text()
    assert next(store.glob("wheels/*/fixture_base*.whl")).read_bytes() == (
        b"base wheel bytes; execution is covered by generation tests"
    )


def test_corrupt_manifest_and_wheel_fall_back_across_configured_mirrors(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    _, engine, channel = _mirror(first)
    _mirror(second)
    # Reuse the exact same base archive/manifest since archive part identity is not deterministic.
    import shutil

    shutil.rmtree(second)
    shutil.copytree(first, second)
    release = channel["engine"]["release"][7:]
    (first / f"engine/{release}/release-manifest.json").write_bytes(b"corrupt")
    (first / f"engine/{release}/fixture_engine-2.0-py3-none-any.whl").write_bytes(b"short")
    catalog = discover((str(first), str(second)), "main")
    assert catalog.engine == engine
    acquire(catalog, "fixture-cpu", tmp_path / "store")
    assert (
        next((tmp_path / "store/wheels").glob("*/fixture_engine*.whl")).read_bytes()
        == b"engine wheel bytes"
    )
    retained = (
        object_path(tmp_path / "store/engines", "sha256:" + release) / "release-manifest.json"
    )
    assert (
        retained.read_bytes() == (second / f"engine/{release}/release-manifest.json").read_bytes()
    )


def test_refuses_protocol_mismatch_before_creating_a_store(tmp_path: Path):
    _mirror(tmp_path / "mirror", protocol=12)
    with pytest.raises(EnvironmentError, match="workerProtocol mismatch"):
        discover((str(tmp_path / "mirror"),), "main")


def test_unsigned_external_channel_is_refused_before_network(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        pytest.fail("untrusted channel attempted network access")

    monkeypatch.setattr("dinkster_supervisor.environment_artifacts.read_metadata", fail)
    with pytest.raises(EnvironmentError, match="valid trusted channel"):
        discover(("https://not-dinkster.example",), "stable")


def test_part_hashes_do_not_substitute_for_verifying_extracted_wheel_hashes(tmp_path: Path):
    base, _, _ = _mirror(tmp_path / "mirror", damaged_archive=True)
    catalog = discover((str(tmp_path / "mirror"),), "main")
    store = tmp_path / "store"
    with pytest.raises(EnvironmentError, match="extracted base wheel differs"):
        acquire(catalog, "fixture-cpu", store)
    assert not object_path(store / "bases", base["id"]).exists()
    assert not list(store.glob("wheels/*/*.whl"))


@pytest.mark.parametrize("labelled", [False, True])
def test_channel_builder_round_trips_both_channels_without_changing_publisher_inputs(
    tmp_path: Path,
    labelled: bool,
):
    mirror = tmp_path / "mirror"
    base, engine, channel = _mirror(mirror, variant="linux-cpu")
    second = tmp_path / "second"
    extra, _, _ = _mirror(second, variant="win-cpu")
    shutil.copytree(second / "base/win-cpu", mirror / "base/win-cpu")
    release = mirror / "engine" / channel["engine"]["release"][7:]
    bases = [
        mirror / "base" / variant / manifest["id"][7:] / "manifest.json"
        for variant, manifest in (("linux-cpu", base), ("win-cpu", extra))
    ]
    if labelled:
        engine.pop("tag")
        engine.update(label="2026-10-31-0.4.0.dev1", builtAt="2026-10-31T12:00:00Z")
        data = _json(engine) + b"\n"
        (release / "release-manifest.json").write_bytes(data)
        name = f"{engine['label']}-{hashlib.sha256(data).hexdigest()[:8]}"
        target = release.with_name(name)
        release.rename(target)
        release = target
    before = {
        path: path.read_bytes()
        for tree in (mirror / "engine", mirror / "base")
        for path in tree.rglob("*")
        if path.is_file()
    }
    expected_id = (
        "sha256:" + hashlib.sha256((release / "release-manifest.json").read_bytes()).hexdigest()
    )
    for name in ("main", "stable"):
        command = [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment_channels",
            "--release",
            str(release),
            "--channel",
            name,
            "--output-root",
            str(mirror),
            "--minimum-desktop-version",
            "0.7.2",
        ]
        for path in reversed(bases):
            command.extend(["--base-manifest", str(path)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        generated = json.loads((mirror / "channels" / f"{name}.json").read_bytes())
        assert generated["engine"]["release"] == expected_id
        assert generated["engine"]["tag"] == (
            (f"v{engine['version']}" if name == "stable" else f"main-{release.name}")
            if labelled
            else engine["tag"]
        )
        assert generated["minimumDesktopVersion"] == "0.7.2"
        selected = discover((str(mirror),), name)
        assert selected.engine == engine
        assert selected.bases == {"linux-cpu": base, "win-cpu": extra}
        assert selected.channel.base_for("linux-cpu") == base["id"]
        assert selected.channel.base_for("win-cpu") == extra["id"]
        assert {path: path.read_bytes() for path in before} == before
    if not labelled:
        assert (mirror / "channels/main.json").read_bytes() == (
            mirror / "channels/stable.json"
        ).read_bytes()


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("constraints", "artifact constraints.txt differs"),
        ("wheel", "artifact fixture_engine.* differs"),
        ("base-id", "base manifest digest mismatch"),
        ("protocol", "workerProtocol mismatch"),
        ("version", "disagrees with engine lock"),
        ("duplicate", "duplicate selected base variant"),
        ("alias", "alias tags"),
        ("empty", "at least one base"),
    ],
)
def test_channel_builder_refuses_invalid_inputs_without_replacing_channels(
    tmp_path: Path, damage: str, reason: str
):
    mirror = tmp_path / "mirror"
    base, engine, channel = _mirror(mirror)
    release = mirror / "engine" / channel["engine"]["release"][7:]
    base_path = mirror / "base/fixture-cpu" / base["id"][7:] / "manifest.json"
    selected = [base_path]
    if damage == "constraints":
        (release / "constraints.txt").write_bytes(b"tampered constraints")
    elif damage == "wheel":
        wheel = release / "fixture_engine-2.0-py3-none-any.whl"
        wheel.write_bytes(b"x" * wheel.stat().st_size)
    elif damage == "base-id":
        base["python"] = "3.14.99"
        base_path.write_bytes(_json(base))
    elif damage == "protocol":
        engine["workerProtocol"] += 1
        (release / "release-manifest.json").write_bytes(_json(engine))
    elif damage == "version":
        constraints = (
            (release / "constraints.txt")
            .read_bytes()
            .replace(b"fixture-base==1.0", b"fixture-base==9.0")
        )
        (release / "constraints.txt").write_bytes(constraints)
        engine["artifacts"][0] = _artifact("constraints.txt", constraints)
        (release / "release-manifest.json").write_bytes(_json(engine))
    elif damage == "duplicate":
        selected.append(base_path)
    elif damage == "alias":
        engine["tag"] = "latest"
        (release / "release-manifest.json").write_bytes(_json(engine))
    else:
        selected.clear()
    before = {path: path.read_bytes() for path in (mirror / "channels").iterdir()}
    for output in (mirror, tmp_path / "absent"):
        with pytest.raises(EnvironmentError, match=reason):
            write_channel(
                release,
                selected,
                channel="main",
                output_root=output,
                minimum_desktop_version="0.2.0",
            )
        assert {path: path.read_bytes() for path in before} == before
        assert not (tmp_path / "absent").exists()


def test_custom_rebuild_preserves_recoverable_archive_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from dinkster_supervisor import environment_custom

    mirror, store = tmp_path / "mirror", tmp_path / "store"
    template, _, _ = _mirror(mirror)
    sources = (str(mirror / "base" / template["variant"] / template["id"][7:]),)
    definition = tmp_path / "variants.toml"
    definition.write_text(
        '[variants.studio-cuda]\ntorch = "2.14.1"\n'
        'torch-index = "https://download.pytorch.org/whl/cu130"\n'
    )
    selection = custom_selection(definition, "studio-cuda", template)
    assert selection is not None
    store.mkdir()

    def resolve(command: list[str], *, cwd: Path) -> str:
        (cwd / "uv.lock").write_text(
            '[[package]]\nname = "fixture-base"\nversion = "1.0"\n'
            '[package.source]\nregistry = "https://pypi.org/simple"\n'
        )
        (cwd / "pylock.toml").write_text("packages = []\n")
        return ""

    monkeypatch.setattr(environment_custom, "run_local_command", resolve)
    monkeypatch.setattr(
        environment_custom,
        "retain_wheels",
        lambda *_: json.loads(json.dumps(template["wheels"])),
    )
    first = environment_custom.build_custom_base(selection, template, sources, store)
    retained = object_path(store / "downloads", first["id"])
    original = {path.name: path.read_bytes() for path in retained.iterdir()}
    assert environment_custom.build_custom_base(selection, template, sources, store) == first
    assert {path.name: path.read_bytes() for path in retained.iterdir()} == original

    download_only = tmp_path / "download-only"
    shutil.copytree(store / "downloads", download_only / "downloads")
    assert (
        environment_custom.build_custom_base(selection, template, sources, download_only) == first
    )
    assert {
        path.name: path.read_bytes()
        for path in object_path(download_only / "downloads", first["id"]).iterdir()
    } == original
    recovered = environment_artifacts.acquire_base(first, (str(retained),), tmp_path / "recovered")
    for wheel in first["wheels"]:
        assert (
            hashlib.sha256((recovered / "wheels" / wheel["filename"]).read_bytes()).hexdigest()
            == (wheel["sha256"])
        )
