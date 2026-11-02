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
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from platform import machine
from types import SimpleNamespace

import pytest
import zstandard
from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_supervisor import environment_artifacts
from dinkster_supervisor.environment_artifacts import (
    acquire,
    acquire_pack_locks,
    acquire_sources,
    discover,
)
from dinkster_supervisor.environment_channels import write_channel
from dinkster_supervisor.environment_custom import custom_selection, variant_file
from dinkster_supervisor.environment_downloads import DownloadError
from dinkster_supervisor.environment_state import EnvironmentError
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
        cached = artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
        assert cached.read_bytes() == payloads[f"{entry.pack}-1.0.zip"]
        cached.write_bytes(b"damaged cache")
    with pytest.raises(EnvironmentError, match="intact cached release source"):
        acquire_sources(catalog, store, download=False)
    for entry in entries:
        cached = artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
        assert cached.read_bytes() == b"damaged cache"
    acquire_sources(catalog, store)
    for entry in entries:
        cached = artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
        assert cached.read_bytes() == payloads[f"{entry.pack}-1.0.zip"]
    target = tmp_path / "target"
    target.mkdir()
    stage_generation_packs(
        target,
        Lockfile.from_record_json(lock.read_text()),
        {
            entry.pack: artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
            for entry in entries
        },
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
        assert path.parent.name == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("source", ["origin", "index"])
@pytest.mark.parametrize("failure", [None, "hash", "size", "missing", "base-hash", "dependency"])
def test_locked_dependencies_use_exact_remote_bytes_and_reuse_the_shared_store(
    tmp_path: Path, source: str, failure: str | None
):
    mirror, store, packages = tmp_path / "release", tmp_path / "store", tmp_path / "packages"
    base, _, channel = _mirror(mirror)
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
        catalog = discover((str(mirror),), "main")
        engine_filename = "fixture_engine-2.0-py3-none-any.whl"
        engine_payload = _wheel_payload("fixture_engine", "2.0", "fixture-base>=1")
        (mirror / "engine" / channel["engine"]["release"][7:] / engine_filename).write_bytes(
            engine_payload
        )
        environment_artifacts.artifact_rows(catalog.engine["artifacts"], "name")[1].update(
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
        catalog.engine["dependencies"] = {
            "fixture-cpu": {"base": base["id"], "wheels": [base_row, dependency]}
        }
        data = _json(catalog.engine)
        environment_artifacts.validate_manifest(
            data, "sha256:" + hashlib.sha256(data).hexdigest(), None
        )
        package_index = origin + "/simple" if source == "index" else None
        if failure == "hash":
            (packages / filename).write_bytes(b"X" * len(payload))
        elif failure == "size":
            (packages / filename).write_bytes(payload + b"extra")
        elif failure == "missing":
            (packages / filename).unlink()
        elif failure == "base-hash":
            base_row["sha256"] = "a" * 64
        if failure is not None:
            with pytest.raises(
                (DownloadError, EnvironmentError),
                match="third-party requires fixture-base>=2; base provides fixture-base 1.0"
                if failure == "dependency"
                else None,
            ):
                acquire(catalog, "fixture-cpu", store, package_index=package_index)
            assert (store / "wheels" / dependency["sha256"] / filename).exists() == (
                failure == "dependency"
            )
            assert not list(store.glob("engines/*/constraints-fixture-cpu.txt"))
            assert not any(".tar.gz" in path for path in requests)
            if failure == "base-hash":
                assert not store.exists()
            return
        constraints = acquire(catalog, "fixture-cpu", store, package_index=package_index)
        assert (store / "wheels" / dependency["sha256"] / filename).read_bytes() == payload
        assert set(constraints.read_text().splitlines()) == {
            "fixture-engine==2.0 --hash=sha256:" + hashlib.sha256(engine_payload).hexdigest(),
            "third-party==7.2 --hash=sha256:" + hashlib.sha256(payload).hexdigest(),
            "fixture-base==1.0 --hash=sha256:" + base_row["sha256"],
        }
        requested = list(requests)
        (packages / filename).unlink()
        (index / "index.html").unlink()
        assert acquire(catalog, "fixture-cpu", store, package_index=package_index) == constraints
        assert requests == requested
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("failure", [None, "missing-base-dependency", "published-hash"])
def test_custom_acquisition_uses_published_target_and_actual_base_ownership(
    tmp_path: Path, failure: str | None
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    published, _, channel = _mirror(mirror)
    catalog = discover((str(mirror),), "main")
    filename = "fixture_engine-2.0-py3-none-any.whl"
    payload = _wheel_payload(
        "fixture_engine",
        "2.0",
        "nvidia-cublas>=13" if failure == "missing-base-dependency" else "fixture-base>=2",
    )
    release = mirror / "engine" / channel["engine"]["release"][7:]
    (release / filename).write_bytes(payload)
    environment_artifacts.artifact_rows(catalog.engine["artifacts"], "name")[1].update(
        _artifact(filename, payload)
    )
    fields = ("name", "version", "filename", "size", "sha256", "index", "url")
    published_row = {key: published["wheels"][0][key] for key in fields}
    if failure == "published-hash":
        published_row["sha256"] = "a" * 64
    catalog.engine["dependencies"] = {
        "fixture-cpu": {
            "base": published["id"],
            "wheels": [
                published_row,
                {
                    **_artifact("torch-2.14.1+cpu-py3-none-any.whl", b"unused", key="filename"),
                    "name": "torch",
                    "version": "2.14.1+cpu",
                    "index": "https://unreachable.invalid/simple",
                    "url": "https://unreachable.invalid/torch-2.14.1+cpu-py3-none-any.whl",
                },
            ],
        }
    }
    custom = json.loads(_json(published))
    custom.update(variant="my-custom-base", replaces=["fixture-base", "torch"])
    custom["wheels"] = [
        {
            **published["wheels"][0],
            **_artifact(name, data, key="filename"),
            "name": distribution,
            "version": version,
            "lockedVersion": None,
        }
        for distribution, version, name, data in (
            ("fixture-base", "2.0", "fixture_base-2.0-py3-none-any.whl", b"custom dependency"),
            ("torch", "2.15.0+vendor", "torch-2.15.0+vendor-py3-none-any.whl", b"custom torch"),
        )
    ]
    identity = {key: custom[key] for key in environment_artifacts._BASE_IDENTITY}
    identity["archive"] = {"format": "tar+zstd"}
    custom.update(id="sha256:" + hashlib.sha256(_json(identity)).hexdigest(), source="user-defined")
    cached = store / "bases" / custom["id"][7:] / "wheels"
    cached.mkdir(parents=True)
    for row, data in zip(custom["wheels"], (b"custom dependency", b"custom torch"), strict=True):
        (cached / row["filename"]).write_bytes(data)
    before = _json(catalog.engine), _json(published), _json(custom)
    if failure:
        with pytest.raises(
            EnvironmentError,
            match="nvidia-cublas absent" if failure == "missing-base-dependency" else "differs",
        ):
            acquire(catalog, "fixture-cpu", store, base=custom, base_sources=())
        assert not list(store.glob("engines/*/constraints-fixture-cpu-*.txt"))
    else:
        constraints = acquire(catalog, "fixture-cpu", store, base=custom, base_sources=())
        assert constraints.name == f"constraints-fixture-cpu-{custom['id'][7:]}.txt"
        assert set(constraints.read_text().splitlines()) == {
            "fixture-engine==2.0 --hash=sha256:" + hashlib.sha256(payload).hexdigest(),
            *(
                f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}"
                for row in custom["wheels"]
            ),
        }
        assert not list(store.glob("engines/*/constraints-fixture-cpu.txt"))
        with pytest.raises(EnvironmentError, match="paired"):
            acquire(catalog, "fixture-cpu", store, base=custom)
        with pytest.raises(EnvironmentError, match="paired"):
            acquire(catalog, "fixture-cpu", store, base_sources=())
    assert (_json(catalog.engine), _json(published), _json(custom)) == before


@pytest.mark.parametrize("damage", [None, "base", "hash", "removed"])
def test_modern_channel_uses_exact_variant_metadata_not_generic_provenance(
    tmp_path: Path, damage: str | None
):
    mirror = tmp_path / "mirror"
    base, engine, channel = _mirror(mirror)
    release = mirror / "engine" / channel["engine"]["release"][7:]
    base["wheels"][0]["lockedVersion"] = "9.0"
    identity = {key: base[key] for key in environment_artifacts._BASE_IDENTITY}
    identity["archive"] = {"format": "tar+zstd"}
    base["id"] = "sha256:" + hashlib.sha256(_json(identity)).hexdigest()
    base_path = tmp_path / "base.json"
    base_path.write_bytes(_json(base))
    constraints = (
        (release / "constraints.txt")
        .read_bytes()
        .replace(b"fixture-base==1.0", b"fixture-base==9.0")
    )
    (release / "constraints.txt").write_bytes(constraints)
    engine["artifacts"][0] = _artifact("constraints.txt", constraints)
    fields = ("name", "version", "filename", "size", "sha256", "index", "url")
    row = {key: base["wheels"][0][key] for key in fields}
    selected = {"base": base["id"], "wheels": [row]}
    if damage == "base":
        selected["base"] = "sha256:" + "0" * 64
    elif damage == "hash":
        row["sha256"] = "0" * 64
    elif damage == "removed":
        selected["wheels"].append(
            {
                **row,
                "name": "nvidia-cublas",
                "version": "13.0",
                "filename": "nvidia_cublas-13.0-py3-none-any.whl",
                "url": "https://example.org/nvidia_cublas-13.0-py3-none-any.whl",
            }
        )
    engine["dependencies"] = {"fixture-cpu": selected}
    (release / "release-manifest.json").write_bytes(_json(engine))
    before = (mirror / "channels/main.json").read_bytes()
    if damage:
        with pytest.raises(EnvironmentError, match="different base|differs|removed"):
            write_channel(
                release,
                [base_path],
                channel="main",
                output_root=mirror,
                minimum_desktop_version="0.2.0",
            )
        assert (mirror / "channels/main.json").read_bytes() == before
    else:
        write_channel(
            release,
            [base_path],
            channel="main",
            output_root=mirror,
            minimum_desktop_version="0.2.0",
        )
        assert (
            json.loads((mirror / "channels/main.json").read_bytes())["base"]["fixture-cpu"]["id"]
            == base["id"]
        )


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


@pytest.mark.parametrize("target", [("windows", "arm64"), ("linux", "arm64"), ("macos", "riscv64")])
def test_discovery_rejects_unsupported_manifest_architectures(
    tmp_path: Path, target: tuple[str, str]
):
    mirror = tmp_path / "mirror"
    _mirror(mirror, target=target)
    with pytest.raises(EnvironmentError, match="intact expected artifact manifest"):
        discover((str(mirror),), "main")


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


def test_refuses_protocol_mismatch_before_creating_a_store(tmp_path: Path):
    _mirror(tmp_path / "mirror", protocol=12)
    with pytest.raises(EnvironmentError, match="workerProtocol mismatch"):
        discover((str(tmp_path / "mirror"),), "main")


def test_base_identity_cannot_ignore_changed_replacement_membership(tmp_path: Path):
    base, _, _ = _mirror(tmp_path / "mirror")
    path = tmp_path / "mirror/base/fixture-cpu" / base["id"][7:] / "manifest.json"
    base["replaces"] = ["fixture-base"]
    path.write_bytes(_json(base))
    with pytest.raises(EnvironmentError, match="intact expected artifact manifest"):
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
    assert not (store / "bases" / base["id"][7:]).exists()
    assert not list(store.glob("wheels/*/*.whl"))


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "base",
        "engine",
        "github",
        "head",
        "config",
        "event",
        "superseded",
        "identical",
        "absent",
        "diverged",
        "compare",
        "unknown",
        "channel",
        "identical-bytes",
        "upload-recover",
        "upload-exhaust",
        "upload-forbidden",
        "upload-corrupt",
        "channel-recover",
    ],
)
def test_channel_publication_exposes_only_verified_payloads(
    tmp_path: Path, monkeypatch, request, capsys, failure
):
    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import ClientError

    from scripts import publish_environment_channel as publisher

    source, public = tmp_path / "source", tmp_path / "public"
    base, engine, channel = _mirror(source)
    release = source / "engine" / channel["engine"]["release"][7:]
    (release / "SHA256SUMS").write_bytes(b"fixture checksums")
    base_path = source / "base/fixture-cpu" / base["id"][7:] / "manifest.json"
    shutil.copytree(source / "base", public / "base")
    (public / "channels").mkdir()
    stable = b"stable channel must not change"
    (public / "channels/stable.json").write_bytes(stable)
    write_channel(
        release, [base_path], channel="main", output_root=public, minimum_desktop_version="0.2.0"
    )
    published_commit = engine["commit"]
    if failure not in {"identical", "identical-bytes"}:
        published_engine = {**engine, "commit": "e" * 40}
        published_commit = published_engine["commit"]
        data = _json(published_engine)
        identity = hashlib.sha256(data).hexdigest()
        target = public / f"engine/{identity}/release-manifest.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(data)
        document = json.loads((public / "channels/main.json").read_bytes())
        document["engine"]["release"] = "sha256:" + identity
        document["engine"]["locations"] = [
            {"type": "mirror", "path": f"engine/{identity}/release-manifest.json"}
        ]
        (public / "channels/main.json").write_bytes(_json(document))
    elif failure == "identical-bytes":
        document = json.loads((public / "channels/main.json").read_bytes())
        document["minimumDesktopVersion"] = "0.1.0"
        (public / "channels/main.json").write_bytes(_json(document))
    previous = (public / "channels/main.json").read_bytes()
    if failure == "absent":
        (public / "channels/main.json").unlink()

    class MirrorHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if failure == "channel" and self.path == "/channels/main.json":
                self.send_error(403)
                return
            if self.headers.get("User-Agent", "").startswith("Python-urllib/"):
                self.send_error(403)
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(MirrorHandler, directory=str(public)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def cleanup():
        server.shutdown()
        thread.join()
        server.server_close()

    request.addfinalizer(cleanup)
    for name, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": engine["commit"],
        "R2_ACCESS_KEY_ID": "test",
        "R2_SECRET_ACCESS_KEY": "test",
        "R2_ACCOUNT_ID": "test",
        "R2_BUCKET": "test",
        "R2_PUBLIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
    }.items():
        monkeypatch.setenv(name, value)
    if failure == "config":
        monkeypatch.delenv("R2_SECRET_ACCESS_KEY")
    elif failure == "event":
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    elif failure == "base":
        (public / base_path.relative_to(source)).write_bytes(b"wrong base manifest")
    original_urlopen = publisher.urlopen

    def published_input(url, **kwargs):
        if url.startswith("https://api.github.com/"):
            assert url == (
                "https://api.github.com/repos/Kosinkadink/Dinkster/compare/"
                f"{published_commit}...{engine['commit']}"
            )
            if failure == "compare":
                raise ValueError("compare unavailable")
            status = (
                "identical"
                if published_commit == engine["commit"]
                else {"superseded": "behind", "diverged": "diverged", "unknown": "unknown"}.get(
                    failure, "ahead"
                )
            )
            return io.BytesIO(json.dumps({"status": status}).encode())
        if url.startswith("https://github.com/"):
            return io.BytesIO(
                b"wrong"
                if failure == "github"
                else (release / "release-manifest.json").read_bytes()
            )
        return original_urlopen(url, **kwargs)

    attempts = {}
    active = set()
    sleeps = []

    def backoff(delay):
        assert not active, "application retry overlapped an unfinished object transfer"
        sleeps.append(delay)

    monkeypatch.setattr(publisher.time, "sleep", backoff)

    class ObjectStore:
        def head_object(self, *, Bucket, Key):
            if failure == "head":
                raise ClientError({"ResponseMetadata": {"HTTPStatusCode": 403}}, "HeadObject")
            if not (public / Key).exists():
                raise ClientError({"ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")

        def upload_file(self, filename, bucket, key, **kwargs):
            if failure in {"identical", "identical-bytes"} and key == "channels/main.json":
                pytest.fail("identical engine must not upload the channel")
            assert key not in active
            active.add(key)
            try:
                attempts[key] = attempts.get(key, 0) + 1
                config = kwargs["Config"]
                assert config.max_concurrency == 1 and config.use_threads is False
                affected = (
                    failure
                    in {"upload-recover", "upload-exhaust", "upload-forbidden", "upload-corrupt"}
                    and key.endswith("/constraints.txt")
                ) or (failure == "channel-recover" and key == "channels/main.json")
                if affected and (
                    attempts[key] < 3 or failure in {"upload-exhaust", "upload-forbidden"}
                ):
                    error = ClientError(
                        {
                            "ResponseMetadata": {
                                "HTTPStatusCode": 403 if failure == "upload-forbidden" else 503
                            },
                            "Error": {
                                "Code": "AccessDenied"
                                if failure == "upload-forbidden"
                                else "ServiceUnavailable",
                                "Message": (
                                    "Reduce your concurrent request rate for the same object."
                                ),
                            },
                        },
                        "PutObject",
                    )
                    raise S3UploadFailedError("fixture upload failed") from error
            finally:
                active.remove(key)
            target = public / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, target)
            if failure in {"engine", "upload-corrupt"} and target.name == "constraints.txt":
                target.write_bytes(b"corrupted uploaded engine bytes")

    monkeypatch.setattr(publisher, "urlopen", published_input)
    monkeypatch.setattr(publisher.boto3, "client", lambda *args, **kwargs: ObjectStore())
    if failure in {
        "base",
        "engine",
        "github",
        "head",
        "event",
        "compare",
        "unknown",
        "channel",
        "identical-bytes",
        "upload-exhaust",
        "upload-forbidden",
        "upload-corrupt",
    }:
        from urllib.error import HTTPError

        with pytest.raises((ValueError, ClientError, HTTPError, S3UploadFailedError)):
            publisher.publish(release, [base_path], "main", "0.2.0")
    else:
        publisher.publish(release, [base_path], "main", "0.2.0")
    assert (public / "channels/stable.json").read_bytes() == stable
    if failure in {
        "upload-recover",
        "upload-exhaust",
        "upload-forbidden",
        "upload-corrupt",
        "channel-recover",
    }:
        key = (
            "channels/main.json"
            if failure == "channel-recover"
            else next(key for key in attempts if key.endswith("/constraints.txt"))
        )
        assert attempts[key] == {"upload-exhaust": 4, "upload-forbidden": 1}.get(failure, 3)
        if failure == "upload-forbidden":
            assert sleeps == []
        else:
            assert sleeps[:2] == [1, 2]
        if failure == "upload-exhaust":
            assert sleeps == [1, 2, 4]
        if failure in {"upload-exhaust", "upload-forbidden", "upload-corrupt"}:
            assert "channels/main.json" not in attempts
    if failure not in {None, "identical", "absent", "upload-recover", "channel-recover"}:
        assert (public / "channels/main.json").read_bytes() == previous
        if failure in {"superseded", "diverged"}:
            output = capsys.readouterr().out
            assert published_commit in output
            assert engine["commit"] in output
    else:
        selected = discover((str(public),), "main")
        assert selected.engine == engine
        assert selected.bases == {"fixture-cpu": base}
        before = {path: path.read_bytes() for path in public.rglob("*") if path.is_file()}
        published_commit = engine["commit"]
        publisher.publish(release, [base_path], "main", "0.2.0")
        assert {path: path.read_bytes() for path in before} == before


def test_channel_builder_round_trips_both_channels_without_changing_publisher_inputs(
    tmp_path: Path,
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
        assert generated["engine"]["tag"] == engine["tag"]
        assert generated["minimumDesktopVersion"] == "0.7.2"
        selected = discover((str(mirror),), name)
        assert selected.engine == engine
        assert selected.bases == {"linux-cpu": base, "win-cpu": extra}
        assert selected.channel.base_for("linux-cpu") == base["id"]
        assert selected.channel.base_for("win-cpu") == extra["id"]
        assert {path: path.read_bytes() for path in before} == before
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


def test_custom_variant_file_precedence_and_optional_extras_index(tmp_path: Path):
    root, store = tmp_path / "project", tmp_path / "store"
    assert variant_file(root, store, None) is None
    assert not root.exists() and not store.exists()
    root.mkdir()
    store.mkdir()
    definition = (
        "[variants.custom-cuda]\n"
        'torch = "2.14.1+cu130"\n'
        'torch-index = "https://download.pytorch.org/whl/cu130"\n'
    )
    fallback = store / "variants.toml"
    fallback.write_text(definition)
    assert variant_file(root, store, None) == fallback
    project = root / "variants.toml"
    project.write_text(definition)
    assert variant_file(root, store, None) == project
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(definition + 'extras-index = "https://extras.example.org/simple"\n')
    assert variant_file(root, store, explicit) == explicit
    template = {"platform": {"os": "linux", "arch": "x86_64"}}
    selected = custom_selection(explicit, "custom-cuda", template)
    assert selected is not None
    assert selected["torch"] == "2.14.1+cu130"
    assert selected["torch-index"] == "https://download.pytorch.org/whl/cu130"
    assert selected["extras-index"] == "https://extras.example.org/simple"
    assert selected["sourceFile"] == str(explicit)
    assert (
        selected["definitionDigest"]
        == "sha256:" + hashlib.sha256(explicit.read_bytes()).hexdigest()
    )
    default = custom_selection(project, "custom-cuda", template)
    assert default is not None and default["extras-index"] == "https://pypi.org/simple"
    assert custom_selection(project, "published-cpu", template) is None
    assert (
        explicit.read_text() == definition + 'extras-index = "https://extras.example.org/simple"\n'
    )


@pytest.mark.parametrize(
    "extra",
    [
        'torch = ">=2.14"',
        'torch-index = "http://download.pytorch.org/whl/cu130"',
        'extras-index = "https://user:password@example.org/simple"',
        'os = "windows"',
    ],
)
def test_custom_variant_rejects_invalid_definitions_before_store_creation(
    tmp_path: Path, extra: str
):
    row = {
        "torch": 'torch = "2.14.1"',
        "torch-index": 'torch-index = "https://download.pytorch.org/whl/cu130"',
    }
    row[extra.split(" =")[0]] = extra
    definition = tmp_path / "variants.toml"
    definition.write_text("[variants.custom-cuda]\n" + "\n".join(row.values()))
    with pytest.raises(EnvironmentError):
        custom_selection(definition, "custom-cuda", {"platform": {"os": "linux", "arch": "x86_64"}})
    assert set(tmp_path.iterdir()) == {definition}


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
    retained = store / "downloads" / first["id"][7:]
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
        for path in (download_only / "downloads" / first["id"][7:]).iterdir()
    } == original
    recovered = environment_artifacts.acquire_base(first, (str(retained),), tmp_path / "recovered")
    for wheel in first["wheels"]:
        assert (
            hashlib.sha256((recovered / "wheels" / wheel["filename"]).read_bytes()).hexdigest()
            == (wheel["sha256"])
        )
